/*
 * -----------------------------------------------------------------------------
 * Firmware Final para Nodo Sensor de Video (ESP32-CAM)
 *
 * [VERSIÓN FINAL 1.1 - CORRECCIÓN DE COMPILACIÓN]
 * - Se han añadido las definiciones de los pines de la cámara directamente en
 * el código para resolver los errores de compilación en un nuevo PC.
 * - El código ahora es autocontenido y no depende de archivos externos.
 *
 * Autor: Félix Gracia (Adaptado por Asistente)
 * Fecha: 21 de Agosto de 2025
 * -----------------------------------------------------------------------------
 */

#define MQTT_MAX_PACKET_SIZE 20480

#include "esp_camera.h"
#include <WiFi.h>
#include <PubSubClient.h>
#include "base64.h"

// --- CONFIGURACIÓN WIFI Y MQTT ---
// Reemplaza con tus credenciales
const char* WIFI_SSID = "H2024";
const char* WIFI_PASS = "FELIX2024";
const char* MQTT_BROKER_IP = "192.168.1.13"; // <-- IP de tu PC
const int MQTT_PORT = 1883;
const char* DEVICE_ID = "ESP32-CAM-Estudiante-03-VIDEO";
const char* MQTT_TOPIC_VIDEO = "institucion/aula/3B/est_03/video";

// --- CORRECCIÓN CLAVE: Definición de pines para el modelo AI-THINKER ---
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// --- Variables Globales ---
WiFiClient espClient;
PubSubClient mqttClient(espClient);
unsigned long lastMsg = 0;
const int msgInterval = 100; // Enviar una imagen cada 5 segundos

// --- Declaración de funciones ---
void reconnect_mqtt();
void capture_and_send_image();

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 10000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_UXGA; // Inicia con alta resolución
  config.jpeg_quality = 20;
  config.fb_count = 1;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;

  // Configuración avanzada del ejemplo de Arduino
  if (psramFound()) {
    config.jpeg_quality = 15;
    config.fb_count = 2;
    config.grab_mode = CAMERA_GRAB_LATEST;
  } else {
    config.frame_size = FRAMESIZE_SVGA;
    config.fb_location = CAMERA_FB_IN_DRAM;
  }

  // Iniciar la cámara
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Fallo al iniciar la cámara! Error 0x%x\n", err);
    return;
  }
  Serial.println("Cámara iniciada correctamente.");

  // Bajar la resolución para el envío, como en el ejemplo
  sensor_t * s = esp_camera_sensor_get();
  s->set_framesize(s, FRAMESIZE_QVGA);

  // Conectar a WiFi
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  WiFi.setSleep(false);
  Serial.print("Conectando a WiFi...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi conectado!");
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());

  // Configurar MQTT
  mqttClient.setServer(MQTT_BROKER_IP, MQTT_PORT);
  mqttClient.setBufferSize(MQTT_MAX_PACKET_SIZE);
  Serial.println("Cliente MQTT configurado.");
}

void loop() {
  if (!mqttClient.connected()) {
    reconnect_mqtt();
  }
  mqttClient.loop();

  unsigned long now = millis();
  if (now - lastMsg > msgInterval) {
    lastMsg = now;
    Serial.printf("Memoria libre: %d bytes\n", ESP.getFreeHeap());
    capture_and_send_image();
    Serial.println("------------------------------------");
  }
  delay(1);
}

void reconnect_mqtt() {
  while (!mqttClient.connected()) {
    Serial.print("Intentando conexión MQTT...");
    if (mqttClient.connect(DEVICE_ID)) {
      Serial.println("conectado!");
    } else {
      Serial.print("falló, rc=");
      Serial.print(mqttClient.state());
      Serial.println(" intentando de nuevo en 5 segundos");
      delay(5000);
    }
  }
}

void capture_and_send_image() {
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Fallo en la captura de la cámara");
    return;
  }

  char json_header[200];
  sprintf(json_header, "{\"timestamp\":\"%lu\",\"device_id\":\"%s\",\"payload_type\":\"video\",\"format\":\"jpeg_base64\",\"data\":\"", (unsigned long)time(NULL), DEVICE_ID);
  
  size_t header_len = strlen(json_header);
  size_t base64_len = ((fb->len + 2) / 3) * 4;
  size_t footer_len = 2; // "}"
  size_t total_len = header_len + base64_len + footer_len;

  if (mqttClient.beginPublish(MQTT_TOPIC_VIDEO, total_len, false)) {
    mqttClient.print(json_header);

    const int CHUNK_SIZE = 1023;
    uint8_t *input_buffer = fb->buf;
    size_t input_len = fb->len;

    for (size_t i = 0; i < input_len; i += CHUNK_SIZE) {
      size_t current_chunk_size = (i + CHUNK_SIZE < input_len) ? CHUNK_SIZE : input_len - i;
      String chunk_encoded = base64::encode(input_buffer + i, current_chunk_size);
      mqttClient.print(chunk_encoded);
    }

    mqttClient.print("\"}");
    
    if (mqttClient.endPublish()) {
      Serial.println("-> Mensaje de video publicado exitosamente.");
    } else {
      Serial.println("-> Fallo al finalizar la publicación de video.");
    }
  } else {
    Serial.println("-> Fallo al iniciar la publicación de video.");
  }

  esp_camera_fb_return(fb);
}
