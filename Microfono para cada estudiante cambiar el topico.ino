/*
 * -----------------------------------------------------------------------------
 * Firmware del Nodo Sensor de AUDIO para ESP32-WROOM con INMP441
 *
 * [VERSIÓN 2.4 - AMPLIFICADO Y BAJA LATENCIA]
 * - Se añade un factor de amplificación digital (AMPLIFICATION_FACTOR) para
 * aumentar la sensibilidad del micrófono.
 * - Se reduce el tamaño del búfer de envío para minimizar la latencia y
 * lograr un streaming más cercano al tiempo real.
 *
 * Autor: Félix Gracia (Revisado y optimizado por Asistente Tesis)
 * Fecha: 05 de Septiembre de 2025
 * -----------------------------------------------------------------------------
 */

#define MQTT_MAX_PACKET_SIZE 8192

#include <WiFi.h>
#include <PubSubClient.h>
#include "driver/i2s.h"
#include "base64.h"

// --- CONFIGURACIÓN WIFI Y MQTT ---
const char* WIFI_SSID      = "H2024";
const char* WIFI_PASS      = "FELIX2024";
const char* MQTT_BROKER_IP = "192.168.1.13"; // IP de  PC
const int   MQTT_PORT      = 1883;

// --- IDENTIFICADOR DEL ESTUDIANTE ---
const char* DEVICE_ID        = "ESP32-WROOM-Estudiante-01-AUDIO";
const char* MQTT_TOPIC_AUDIO = "institucion/aula/3B/est_01/audio";

// --- CONFIGURACIÓN DEL MICRÓFONO I2S ---
#define I2S_WS_PIN    25
#define I2S_SCK_PIN   26
#define I2S_SD_PIN    22
#define I2S_PORT      I2S_NUM_0

// --- PARÁMETROS DE AUDIO ---
#define SAMPLE_RATE   16000
#define BUFFER_SIZE   512 
#define AMPLIFICATION_FACTOR 2.5 

// --- Variables Globales ---
WiFiClient espClient;
PubSubClient mqttClient(espClient);

// --- Declaración de funciones ---
void setup_wifi_and_mqtt();
void reconnect_mqtt();
void setup_microphone();
void capture_and_send_audio();

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== Iniciando Nodo de Audio (v2.4 - Amplificado) ===");
  
  setup_wifi_and_mqtt();
  setup_microphone();
}

void loop() {
  if (!mqttClient.connected()) {
    reconnect_mqtt();
  }
  mqttClient.loop();
  capture_and_send_audio();
}

void setup_wifi_and_mqtt() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Conectando a WiFi...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi conectado!");
  Serial.print("Dirección IP: ");
  Serial.println(WiFi.localIP());

  mqttClient.setServer(MQTT_BROKER_IP, MQTT_PORT);
  mqttClient.setBufferSize(MQTT_MAX_PACKET_SIZE);
  Serial.println("Cliente MQTT configurado.");
}

void reconnect_mqtt() {
  while (!mqttClient.connected()) {
    Serial.print("Intentando conexión MQTT...");
    if (mqttClient.connect(DEVICE_ID)) {
      Serial.println(" ¡conectado!");
    } else {
      Serial.print(" falló, rc=");
      Serial.print(mqttClient.state());
      Serial.println(" | Intentando de nuevo en 5 segundos");
      delay(5000);
    }
  }
}

void setup_microphone() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = 0,
    .dma_buf_count = 8,
    .dma_buf_len = 256,
    .use_apll = true
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK_PIN,
    .ws_io_num = I2S_WS_PIN,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD_PIN
  };

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);
  Serial.println("Micrófono I2S iniciado correctamente.");
}

void capture_and_send_audio() {
  int32_t i2s_read_buffer[BUFFER_SIZE];
  int16_t samples_16bit[BUFFER_SIZE];
  size_t bytes_read = 0;
  
  i2s_read(I2S_PORT, &i2s_read_buffer, BUFFER_SIZE * 4, &bytes_read, portMAX_DELAY);
  
  if (bytes_read > 0) {
    int samples_read = bytes_read / 4;
    for (int i = 0; i < samples_read; i++) {
      // Extraer la muestra de 16 bits
      int16_t sample = (i2s_read_buffer[i] >> 16);
      
      // [NUEVO] Aplicar amplificación digital
      int32_t amplified_sample = (int32_t)sample * AMPLIFICATION_FACTOR;

      // [NUEVO] Control de saturación (clipping) para evitar distorsión
      if (amplified_sample > 32767) {
        amplified_sample = 32767;
      } else if (amplified_sample < -32768) {
        amplified_sample = -32768;
      }
      samples_16bit[i] = (int16_t)amplified_sample;
    }
    
    String encodedAudio = base64::encode((uint8_t*)samples_16bit, samples_read * 2);

    String json_payload = "{\"timestamp\":" + String(time(NULL)) + "," +
                          "\"device_id\":\"" + DEVICE_ID + "\"," +
                          "\"payload_type\":\"audio\"," +
                          "\"format\":\"pcm_base64\"," +
                          "\"data\":\"" + encodedAudio + "\"}";

    if (!mqttClient.publish(MQTT_TOPIC_AUDIO, json_payload.c_str())) {
      // El reconnect se gestionará en el bucle principal
    }
  }
}

