/*
 * SafeSound - Wokwi Simulation Sketch
 * =====================================
 * Simplified firmware for Wokwi online simulator.
 * Demonstrates: LED alert patterns, vibration motor, button input.
 *
 * Since Wokwi cannot simulate I2S microphone or TFLite Micro,
 * audio classification is simulated via:
 *   - Button press (cycles through all 6 classes)
 *   - Serial commands (type class name to trigger)
 *
 * Serial commands:
 *   "fire"   -> fire alarm detected
 *   "baby"   -> baby cry detected
 *   "choke"  -> choking detected
 *   "horn"   -> car horn detected
 *   "door"   -> doorbell detected
 *   "clear"  -> return to idle
 *
 * Pin assignments:
 *   GPIO 5  -> WS2812B LED strip (DIN)
 *   GPIO 6  -> Vibration motor (via NPN transistor)
 *   GPIO 7  -> Status LED (green, system ready)
 *   GPIO 4  -> MODE button (reset to idle)
 *   GPIO 8  -> TEST button (cycle classes)
 *   GPIO 9  -> Charging indicator LED
 *   GPIO 14 -> I2S SCK (INMP441, shown connected)
 *   GPIO 15 -> I2S WS  (INMP441, shown connected)
 *   GPIO 13 -> I2S SD  (INMP441, shown connected)
 */

#include <Adafruit_NeoPixel.h>

#define LED_PIN         5
#define NUM_LEDS        8
#define MOTOR_PIN       6
#define STATUS_PIN      7
#define BTN_MODE_PIN    4
#define BTN_TEST_PIN    8
#define CHARGE_PIN      9

#define I2S_SCK_PIN     14
#define I2S_WS_PIN      15
#define I2S_SD_PIN      13

#define NUM_CLASSES     6

Adafruit_NeoPixel strip(NUM_LEDS, LED_PIN, NEO_GRB + NEO_KHZ800);

const char* CLASS_NAMES[] = {
  "FIRE_ALARM", "BABY_CRY", "CHOKING", "CAR_HORN", "DOORBELL", "BACKGROUND"
};

const char* PRIORITY_NAMES[] = {
  "CRITICAL", "HIGH", "CRITICAL", "HIGH", "MEDIUM", "NONE"
};

// RGB colors per class
const uint32_t LED_COLORS[] = {
  0xE63946,  // fire: red
  0xF4A261,  // baby: orange
  0xE63946,  // choking: red
  0xE9C46A,  // horn: yellow
  0x457B9D,  // doorbell: blue
  0x000000,  // background: off
};

// Simulated confidence per class
const float SIM_CONFIDENCE[] = {0.92, 0.89, 0.87, 0.91, 0.92, 0.97};

// Vibration pulses per class
const int VIB_PULSES[] = {5, 3, 5, 3, 1, 0};

// Thresholds
const float THRESHOLDS[] = {0.75, 0.80, 0.75, 0.80, 0.90, 0.50};

// Temporal smoothing
#define SMOOTH_FRAMES 3
float smooth_buf[SMOOTH_FRAMES][NUM_CLASSES];
int smooth_idx = 0;

int current_class = 5; // background
bool alert_active = false;
bool last_test_btn = HIGH;
bool last_mode_btn = HIGH;
unsigned long last_press = 0;

void setup() {
  Serial.begin(115200);
  delay(300);

  strip.begin();
  strip.setBrightness(60);
  strip.show();

  pinMode(MOTOR_PIN, OUTPUT);
  pinMode(STATUS_PIN, OUTPUT);
  pinMode(CHARGE_PIN, OUTPUT);
  pinMode(BTN_MODE_PIN, INPUT_PULLUP);
  pinMode(BTN_TEST_PIN, INPUT_PULLUP);

  // Init smoothing buffer
  for (int f = 0; f < SMOOTH_FRAMES; f++)
    for (int c = 0; c < NUM_CLASSES; c++)
      smooth_buf[f][c] = (c == 5) ? 0.9 : 0.02;

  // Boot sequence
  startup_animation();

  Serial.println("========================================");
  Serial.println("  SafeSound v1.0");
  Serial.println("  ESP32-S3 | DS-CNN INT8 | 36.4 KB");
  Serial.println("========================================");
  Serial.println("Serial: fire, baby, choke, horn, door, clear");
  Serial.println("Buttons: TEST (cycle), MODE (reset)");
  Serial.println("Listening...");
  Serial.println();
}

void startup_animation() {
  for (int i = 0; i < NUM_LEDS; i++) {
    strip.setPixelColor(i, 0x3FB950);
    strip.show();
    delay(60);
  }
  delay(400);
  strip.clear();
  strip.show();
  digitalWrite(STATUS_PIN, HIGH);
}

void trigger_alert(int cls) {
  current_class = cls;
  alert_active = (cls != 5);

  float conf = SIM_CONFIDENCE[cls];

  Serial.println("--------------------------------------------");
  Serial.print("  DETECTED: ");
  Serial.println(CLASS_NAMES[cls]);
  Serial.print("  Confidence: ");
  Serial.print(conf * 100, 1);
  Serial.print("% | Threshold: ");
  Serial.print(THRESHOLDS[cls] * 100, 0);
  Serial.println("%");
  Serial.print("  Priority: ");
  Serial.print(PRIORITY_NAMES[cls]);
  Serial.print(" | Vibration: ");
  Serial.print(VIB_PULSES[cls]);
  Serial.println(" pulses");
  Serial.println("--------------------------------------------");

  if (cls == 5) {
    strip.clear();
    strip.show();
    digitalWrite(MOTOR_PIN, LOW);
    Serial.println("  Status: IDLE (no alert)");
    return;
  }

  // LED alert pattern
  uint32_t color = LED_COLORS[cls];
  uint8_t r = (color >> 16) & 0xFF;
  uint8_t g = (color >> 8) & 0xFF;
  uint8_t b = color & 0xFF;

  for (int blink = 0; blink < 3; blink++) {
    for (int i = 0; i < NUM_LEDS; i++)
      strip.setPixelColor(i, r, g, b);
    strip.show();
    delay(200);
    strip.clear();
    strip.show();
    delay(100);
  }

  // Keep LEDs on
  for (int i = 0; i < NUM_LEDS; i++)
    strip.setPixelColor(i, r, g, b);
  strip.show();

  // Vibration pattern
  int pulses = VIB_PULSES[cls];
  for (int p = 0; p < pulses; p++) {
    digitalWrite(MOTOR_PIN, HIGH);
    delay(80);
    digitalWrite(MOTOR_PIN, LOW);
    delay(60);
  }

  Serial.println("  Alert delivered.");
  Serial.println();
}

void clear_alert() {
  current_class = 5;
  alert_active = false;
  strip.clear();
  strip.show();
  digitalWrite(MOTOR_PIN, LOW);
  Serial.println("[RESET] Idle - listening...");
}

void process_serial() {
  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  cmd.toLowerCase();

  if (cmd == "fire") trigger_alert(0);
  else if (cmd == "baby") trigger_alert(1);
  else if (cmd == "choke") trigger_alert(2);
  else if (cmd == "horn") trigger_alert(3);
  else if (cmd == "door") trigger_alert(4);
  else if (cmd == "clear") clear_alert();
  else {
    Serial.print("Unknown: ");
    Serial.println(cmd);
  }
}

void loop() {
  if (Serial.available()) {
    process_serial();
  }

  // TEST button: cycle through classes
  bool test_btn = digitalRead(BTN_TEST_PIN);
  if (test_btn == LOW && last_test_btn == HIGH && (millis() - last_press > 300)) {
    last_press = millis();
    current_class = (current_class + 1) % NUM_CLASSES;
    trigger_alert(current_class);
  }
  last_test_btn = test_btn;

  // MODE button: reset
  bool mode_btn = digitalRead(BTN_MODE_PIN);
  if (mode_btn == LOW && last_mode_btn == HIGH && (millis() - last_press > 300)) {
    last_press = millis();
    clear_alert();
  }
  last_mode_btn = mode_btn;

  // Breathing animation on status LED (shows system is alive)
  int breath = (millis() / 8) % 512;
  if (breath > 255) breath = 511 - breath;
  analogWrite(STATUS_PIN, breath);

  delay(20);
}
