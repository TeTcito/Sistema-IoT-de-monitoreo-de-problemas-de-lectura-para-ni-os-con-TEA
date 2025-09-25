# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# analizador_con_visualizacion.py - VERSIÓN 30.1 (VERSIÓN FINAL CON VISUALIZACIÓN)
#
# [VERSIÓN 30.1]
# - VISUALIZACIÓN ACTIVADA: Se reincorpora el dibujado de la malla facial
#   de MediaPipe y los puntos del iris para fines de demostración y evidencia.
# - LÓGICA INTACTA: Todas las funcionalidades de análisis y alertas de la v30
#   se mantienen sin cambios, garantizando el mismo rendimiento y precisión.
#
# Autor: Félix Gracia (Versión final por Asistente Tesis)
# Fecha: 06 de Septiembre de 2025
# -----------------------------------------------------------------------------

import sys
import os
import json
import base64
import time
import queue
import re
import threading
from datetime import datetime
import traceback

# --- IMPORTAR LIBRERÍAS REQUERIDAS ---
try:
    from scipy.spatial import distance as dist
    import paho.mqtt.client as mqtt
    import numpy as np
    import cv2
    import firebase_admin
    from firebase_admin import credentials, db
    import mediapipe as mp
    import sounddevice as sd
    from vosk import Model, KaldiRecognizer
    from transformers import pipeline
    import torch
except ImportError as e:
    print(f"❌ Error: Falta una librería esencial -> {e}"); sys.exit()

# --- CONFIGURACIÓN GENERAL Y RUTAS ---
BROKER_IP = "localhost"
BROKER_PORT = 1883
SUSCRIPTION_TOPIC = "institucion/aula/3B/+/+"
CLIENT_ID = f"pc_analizador_main_{int(time.time())}"
AUDIO_DEVICE_ID = 3 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print("📖 Sistema de análisis de habla v30.1 (Versión Final con Visualización) iniciado.")

# --- CONFIGURACIÓN DE FIREBASE ---
try:
    json_key_path = os.path.join(SCRIPT_DIR, "serviceAccountKey.json")
    cred = credentials.Certificate(json_key_path)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://atencion-lectora-default-rtdb.firebaseio.com'
    })
    print("✅ Conexión con Firebase establecida exitosamente.")
except Exception as e:
    print(f"❌ ERROR CRÍTICO AL CONECTAR CON FIREBASE: {e}"); sys.exit()
    
# --- CONFIGURACIÓN DE MODELOS DE IA ---
try:
    model_folder_name = "vosk-model-es-0.42"
    model_path = os.path.join(SCRIPT_DIR, model_folder_name)
    if not os.path.exists(model_path):
        print(f"❌ Error: El modelo de Vosk no se encontró en la ruta '{model_path}'"); sys.exit()
    vosk_model = Model(model_path)
    print("✅ Modelo Lingüista (Vosk) cargado exitosamente.")
except Exception as e:
    print(f"❌ ERROR CRÍTICO AL CARGAR MODELO VOSK: {e}"); sys.exit()

try:
    print("⏳ Cargando modelo Psicólogo (Hugging Face)...")
    prosody_classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")
    print("✅ Modelo Psicólogo (Wav2Vec2) cargado exitosamente.")
except Exception as e:
    print(f"❌ ERROR CRÍTICO AL CARGAR MODELO DE HUGGING FACE: {e}"); sys.exit()

# --- PARÁMETROS DE CALIBRACIÓN Y LÓGICA ---
# Calibración Visual
EAR_THRESHOLD = 0.18
HORIZONTAL_GAZE_THRESHOLD = 0.15
VERTICAL_GAZE_THRESHOLD = 0.12
EYE_CLOSED_CONSEC_FRAMES = 5
FACE_LOSS_THRESHOLD = 15
GAZE_OFF_CONSEC_FRAMES = 5
# Calibración Auditiva
HESITATION_WORDS = {"eh", "este", "pues", "o sea", "bueno", "um", "mm"}
HIGH_HESITATION_RATE = 0.25
HIGH_REPETITION_RATE = 0.20
LOW_WPM_RATE = 80.0
LOW_TTR_RATE = 0.7
LOW_SENTENCE_LEN_RATE = 3.0
PAUSE_THRESHOLD_S = 0.8
AUDIO_ENERGY_THRESHOLD = 300 
LOW_COHERENCE_THRESHOLD = 0.05
SILENCE_THRESHOLD_S = 5.0

# --- ESTRUCTURAS DE DATOS Y ESTADO ---
student_data = {}; data_queue = queue.Queue(); firebase_queue = queue.Queue()
audio_playback_queue = queue.Queue(); running = True; connected = False

# --- PARÁMETROS DE AUDIO ---
SAMPLE_RATE = 16000; CHANNELS = 1

# --- INICIALIZACIÓN DE MEDIAPIPE ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils # <-- AÑADIDO PARA DIBUJAR
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- ÍNDICES DE PUNTOS CLAVE ---
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_IRIS_INDICES = [474, 475, 476, 477]
RIGHT_IRIS_INDICES = [469, 470, 471, 472]

# --- PIPELINE DE ANÁLISIS DE HABLA ---
def extract_linguistic_features(transcribed_text, duration_s):
    text_clean = re.sub(r'[^\w\s]', '', transcribed_text.lower())
    words = text_clean.split()
    num_words = len(words)
    if num_words == 0: return None

    coherence_score = 1.0 
    if num_words > 3:
        bigrams = list(zip(words, words[1:]))
        if bigrams:
            COMMON_SPANISH_BIGRAMS = {
                ('de', 'la'), ('de', 'el'), ('en', 'la'), ('en', 'el'), ('a', 'la'), ('a', 'el'), ('con', 'la'), ('con', 'el'),
                ('por', 'que'), ('para', 'que'), ('que', 'es'), ('lo', 'que'), ('es', 'que'), ('yo', 'soy'), ('yo', 'tengo'),
                ('yo', 'quiero'), ('yo', 'estoy'), ('el', 'es'), ('ella', 'es'), ('es', 'un'), ('es', 'una'), ('me', 'gusta'),
                ('no', 'me'), ('no', 'es'), ('no', 'se'), ('se', 'fue'), ('se', 'ha'), ('voy', 'a'), ('vamos', 'a'),
                ('tiene', 'que'), ('habia', 'una'), ('era', 'un'),
            }
            common_found = sum(1 for bg in bigrams if bg in COMMON_SPANISH_BIGRAMS)
            coherence_score = common_found / len(bigrams)

    return {
        "transcripcion": transcribed_text,
        "ppm": round((num_words / duration_s) * 60 if duration_s > 0 else 0),
        "tasa_hesitacion": round(sum(1 for w in words if w in HESITATION_WORDS) / num_words, 3),
        "tasa_repeticion": round((num_words - len(set(words))) / num_words, 3),
        "riqueza_lexica_ttr": round(len(set(words)) / num_words if num_words > 0 else 0, 3),
        "longitud_frase": num_words,
        "coherence_score": coherence_score
    }

def analyze_prosody(audio_bytes):
    try:
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        result = prosody_classifier({"raw": audio_array, "sampling_rate": SAMPLE_RATE}, top_k=1)
        return result[0]['label'], result[0]['score']
    except Exception: return "N/A", 0.0

def unified_diagnostic_model(linguistic_features, prosody_label):
    if not linguistic_features or linguistic_features["longitud_frase"] < 2:
        return "HABLA MUY CORTA"

    if linguistic_features["longitud_frase"] > 4 and linguistic_features["coherence_score"] < LOW_COHERENCE_THRESHOLD:
         return "LENGUAJE DESESTRUCTURADO"

    if linguistic_features["tasa_repeticion"] > HIGH_REPETITION_RATE:
        return f"DIFICULTAD DE FLUIDEZ (Repeticiones)"
    if linguistic_features["tasa_hesitacion"] > HIGH_HESITATION_RATE:
        return f"DIFICULTAD DE FORMULACIÓN (Muletillas)"
    if linguistic_features["ppm"] < LOW_WPM_RATE and linguistic_features["longitud_frase"] > 4:
        return f"HABLA LENTA (Tono: {prosody_label})"
    if linguistic_features["riqueza_lexica_ttr"] < LOW_TTR_RATE and linguistic_features["longitud_frase"] > 5:
        return f"POSIBLE POBREZA LÉXICA (Tono: {prosody_label})"
    if linguistic_features["longitud_frase"] < LOW_SENTENCE_LEN_RATE:
        return f"HABLA SIMPLIFICADA (Frases cortas)"
    return f"HABLA FLUIDA (Tono: {prosody_label})"

# --- Hilo de trabajo para el análisis ---
def analysis_worker(student_id, audio_to_analyze, recognizer_instance):
    try:
        final_result = json.loads(recognizer_instance.FinalResult())
        transcribed_text = final_result.get('text', '').strip()

        if not transcribed_text:
            return

        duration = 0
        word_info = final_result.get('result', [])
        if word_info:
            start_time = word_info[0]['start']
            end_time = word_info[-1]['end']
            duration = end_time - start_time
        else:
            duration = len(audio_to_analyze) / (SAMPLE_RATE * 2)

        print(f"🎤 Pausa detectada. Analizando frase de {student_id}: '{transcribed_text}' (dur: {duration:.2f}s)")

        linguistic_features = extract_linguistic_features(transcribed_text, duration)
        prosody_label, prosody_score = analyze_prosody(audio_to_analyze)
        final_status = unified_diagnostic_model(linguistic_features, prosody_label.upper())
        
        s_data = student_data.get(student_id)
        if not s_data: return

        s_data['current_speech_status'] = final_status
        s_data['speech_metrics'] = linguistic_features if linguistic_features else {}
        s_data['speech_metrics']['prosody'] = {"label": prosody_label, "score": round(float(prosody_score), 3)}
        
        payload_fb = {
            "attention_status": s_data['current_attention_status'], 
            "speech_status": final_status, 
            "speech_metrics": s_data.get('speech_metrics', {}), 
            "timestamp": datetime.now().isoformat()
        }
        firebase_queue.put((student_id, payload_fb))
        print(f"🗣️ ALERTA DE HABLA {student_id}: A:'{s_data['current_attention_status']}', H:'{final_status}'")
        
        s_data['last_speech_sent'] = final_status

    except Exception as e:
        print(f"⚠️ Error en el hilo de análisis para {student_id}: {e}")

# --- FUNCIONES DE CÁLCULO VISUAL ---
def calculate_ear(eye_landmarks):
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5]); B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3]); return (A + B) / (2.0 * C)

def calculate_gaze_ratios(eye_points, iris_center):
    eye_width = dist.euclidean(eye_points[0], eye_points[3])
    if eye_width == 0: return 0, 0
    
    eye_center_x = (eye_points[0][0] + eye_points[3][0]) / 2
    gaze_ratio_h = (iris_center[0] - eye_center_x) / eye_width

    eye_height = dist.euclidean(eye_points[1], eye_points[4])
    if eye_height == 0: return gaze_ratio_h, 0

    eye_center_y = (eye_points[1][1] + eye_points[5][1]) / 2
    gaze_ratio_v = (iris_center[1] - eye_center_y) / eye_height
    
    return gaze_ratio_h, gaze_ratio_v

# --- HILOS Y MQTT ---
def firebase_thread_func():
    while running:
        try:
            student_id, payload_to_send = firebase_queue.get(timeout=1)
            db.reference(f'analysis/{student_id}').set(payload_to_send)
            db.reference(f'analysis_logs/{student_id}').push(payload_to_send)
        except queue.Empty: continue
        except Exception: pass

def audio_playback_thread():
    stream = None
    try:
        stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', device=AUDIO_DEVICE_ID)
        stream.start()
        while running:
            try:
                _, audio_data = audio_playback_queue.get(timeout=1)
                if running: stream.write(audio_data)
            except queue.Empty: continue
    except Exception: pass
    finally:
        if stream:
            stream.stop(); stream.close()

def on_connect(client, userdata, flags, rc, properties=None):
    global connected
    if rc == 0: client.subscribe(SUSCRIPTION_TOPIC); connected = True; print(f"✅ Analizador conectado al broker.")
    else: print(f"❌ Falló la conexión al broker con código: {rc}")

def on_message(client, userdata, msg):
    try:
        topic_parts = msg.topic.split('/')
        if len(topic_parts) >= 5:
            student_id, data_type = topic_parts[3], topic_parts[4]
            data_queue.put((student_id, data_type, json.loads(msg.payload.decode('utf-8'))))
    except Exception: pass
    
def create_recognizer():
    rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)
    rec.SetWords(True)
    return rec

def initialize_student_data(student_id):
    if student_id not in student_data:
        print(f"✨ Nuevo estudiante detectado: {student_id}. Inicializando estructuras.")
        student_data[student_id] = {
            'id': student_id, 
            'no_face_counter': 0, 
            'eye_closed_counter': 0,
            'gaze_off_counter': 0,
            'current_attention_status': 'Iniciando...', 'last_attention_sent': 'Iniciando...',
            'recognizer': create_recognizer(),
            'last_speech_time': time.time(), 'is_speaking': False,
            'current_speech_status': 'ESCUCHANDO...', 'last_speech_sent': 'ESCUCHANDO...',
            'speech_metrics': {}, 'accumulated_audio': b''
        }
        cv2.namedWindow(f'Video - {student_id}', cv2.WINDOW_NORMAL)

# --- FUNCIÓN PRINCIPAL ---
def main():
    global running
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=CLIENT_ID)
    client.on_connect = on_connect; client.on_message = on_message
    
    try: client.connect(BROKER_IP, BROKER_PORT, 60)
    except Exception as e: print(f"❌ No se pudo conectar al broker. Error: {e}"); return
    
    client.loop_start()
    threading.Thread(target=firebase_thread_func, daemon=True).start()
    threading.Thread(target=audio_playback_thread, daemon=True).start()

    print("⏳ Analizador esperando conexión al broker...")
    time.sleep(2)
    if not connected: print("❌ No se pudo conectar al broker. Terminando."); running = False; return

    print("\n▶️ Análisis multimodal iniciado. Presiona Ctrl+C para salir de forma segura.")
    
    try:
        while running:
            try:
                student_id, data_type, payload = data_queue.get(timeout=0.01)
                initialize_student_data(student_id)
                s_data = student_data.get(student_id)
                if not s_data: continue

                if data_type == 'video':
                    img = cv2.imdecode(np.frombuffer(base64.b64decode(payload['data']), np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        
                        previous_attention_status = s_data['current_attention_status']

                        if results.multi_face_landmarks:
                            s_data['no_face_counter'] = 0
                            face_landmarks_list = results.multi_face_landmarks[0] # <-- AÑADIDO
                            landmarks = np.array([(lm.x * img.shape[1], lm.y * img.shape[0]) for lm in face_landmarks_list.landmark], dtype=np.int32)
                            avg_ear = (calculate_ear(landmarks[LEFT_EYE_INDICES]) + calculate_ear(landmarks[RIGHT_EYE_INDICES])) / 2.0
                            left_iris_center = landmarks[LEFT_IRIS_INDICES].mean(axis=0).astype(np.int32)
                            right_iris_center = landmarks[RIGHT_IRIS_INDICES].mean(axis=0).astype(np.int32)
                            left_gaze_h, left_gaze_v = calculate_gaze_ratios(landmarks[LEFT_EYE_INDICES], left_iris_center)
                            right_gaze_h, right_gaze_v = calculate_gaze_ratios(landmarks[RIGHT_EYE_INDICES], right_iris_center)
                            avg_gaze_h = (left_gaze_h + right_gaze_h) / 2.0
                            avg_gaze_v = (left_gaze_v + right_gaze_v) / 2.0

                            if avg_ear < EAR_THRESHOLD:
                                s_data['eye_closed_counter'] += 1
                                s_data['gaze_off_counter'] = 0
                                if s_data['eye_closed_counter'] >= EYE_CLOSED_CONSEC_FRAMES:
                                    s_data['current_attention_status'] = "OJOS CERRADOS"
                            elif abs(avg_gaze_h) > HORIZONTAL_GAZE_THRESHOLD or abs(avg_gaze_v) > VERTICAL_GAZE_THRESHOLD:
                                s_data['gaze_off_counter'] += 1
                                s_data['eye_closed_counter'] = 0
                                if s_data['gaze_off_counter'] >= GAZE_OFF_CONSEC_FRAMES:
                                    s_data['current_attention_status'] = "SIN ATENCION VISUAL"
                            else:
                                s_data['eye_closed_counter'] = 0
                                s_data['gaze_off_counter'] = 0
                                s_data['current_attention_status'] = "ATENTO"
                            
                            # --- [VISUALIZACIÓN DE PUNTOS CLAVE ACTIVADA] ---
                            mp_drawing.draw_landmarks(
                                image=img, landmark_list=face_landmarks_list,
                                connections=mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=1))
                            cv2.circle(img, tuple(left_iris_center), 3, (0, 0, 255), -1)
                            cv2.circle(img, tuple(right_iris_center), 3, (0, 0, 255), -1)
                            # --- [FIN DE LA VISUALIZACIÓN] ---

                        else:
                            s_data['no_face_counter'] += 1
                            s_data['eye_closed_counter'] = 0
                            s_data['gaze_off_counter'] = 0
                            if s_data['no_face_counter'] >= FACE_LOSS_THRESHOLD: s_data['current_attention_status'] = "DISTRACCION"
                        
                        if s_data['current_attention_status'] != s_data['last_attention_sent']:
                            s_data['last_attention_sent'] = s_data['current_attention_status']
                            payload_fb = {
                                "attention_status": s_data['current_attention_status'], 
                                "speech_status": s_data['current_speech_status'], 
                                "speech_metrics": s_data.get('speech_metrics', {}), 
                                "timestamp": datetime.now().isoformat()
                            }
                            firebase_queue.put((student_id, payload_fb))
                            print(f"👁️ ALERTA VISUAL {student_id}: '{s_data['current_attention_status']}'")

                        cv2.putText(img, f"Atencion: {s_data['current_attention_status']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(img, f"Habla: {s_data['current_speech_status']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2)
                        cv2.imshow(f'Video - {student_id}', img)
                        if cv2.waitKey(1) & 0xFF == ord('q'): running = False

                elif data_type == 'audio':
                    audio_data_bytes = base64.b64decode(payload['data'])
                    audio_playback_queue.put((student_id, np.frombuffer(audio_data_bytes, dtype=np.int16)))
                    s_data['accumulated_audio'] += audio_data_bytes
                    
                    audio_chunk_np = np.frombuffer(audio_data_bytes, dtype=np.int16)
                    rms = np.sqrt(np.mean(audio_chunk_np.astype(float)**2))

                    if rms > AUDIO_ENERGY_THRESHOLD:
                        s_data['is_speaking'] = True
                        s_data['last_speech_time'] = time.time()

                    s_data['recognizer'].AcceptWaveform(audio_data_bytes)

            except queue.Empty:
                pass 

            current_time = time.time()
            for sid, s_data_loop in list(student_data.items()):
                time_since_last_speech = current_time - s_data_loop.get('last_speech_time', 0)
                
                if s_data_loop.get('is_speaking') and time_since_last_speech > PAUSE_THRESHOLD_S:
                    
                    s_data_loop['is_speaking'] = False
                    
                    audio_to_analyze = s_data_loop['accumulated_audio']
                    recognizer_to_analyze = s_data_loop['recognizer']
                    
                    s_data_loop['accumulated_audio'] = b''
                    s_data_loop['recognizer'] = create_recognizer()
                    
                    if len(audio_to_analyze) > SAMPLE_RATE * 0.5:
                        worker = threading.Thread(target=analysis_worker, args=(sid, audio_to_analyze, recognizer_to_analyze), daemon=True)
                        worker.start()
                
                elif not s_data_loop.get('is_speaking') and time_since_last_speech > SILENCE_THRESHOLD_S:
                    if s_data_loop['current_speech_status'] != 'ESCUCHANDO...':
                        print(f"🤫 Silencio prolongado detectado para {sid}. Reseteando estado de habla.")
                        s_data_loop['current_speech_status'] = 'ESCUCHANDO...'
                        s_data_loop['last_speech_sent'] = 'ESCUCHANDO...'
                        s_data_loop['speech_metrics'] = {}

                        payload_fb = {
                            "attention_status": s_data_loop['current_attention_status'],
                            "speech_status": "ESCUCHANDO...",
                            "speech_metrics": {},
                            "timestamp": datetime.now().isoformat()
                        }
                        firebase_queue.put((sid, payload_fb))

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n🛑 Interrupción de teclado detectada. Iniciando apagado seguro...")
    finally:
        running = False
        print("⏳ Esperando a que los hilos finalicen...")
        time.sleep(2)
        client.loop_stop()
        cv2.destroyAllWindows()
        print("✅ Analizador finalizado correctamente.")
        
if __name__ == "__main__":
    main()
