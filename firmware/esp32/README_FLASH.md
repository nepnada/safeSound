# Flashing the ESP32-S3

## 1. Install Arduino IDE 2.x

Download from https://www.arduino.cc/en/software

## 2. Add ESP32 board support

In Arduino IDE → Preferences → Additional boards manager URLs, add:
```
https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
```
Then: Tools → Board → Boards Manager → search "esp32" → Install "esp32 by Espressif Systems"

## 3. Install libraries

Tools → Manage Libraries → install:
- **FastLED** by Daniel Garcia (v3.6+)
- **EloquentTinyML** (for TFLite Micro wrapper)

Or install TFLite Micro manually:
```
https://github.com/espressif/esp-tflite-micro
```

## 4. Generate model_data.h

After training completes:
```bash
cd /Users/Apple/Desktop/iot
source venv/bin/activate
python src/export_model_header.py
```
This creates `firmware/esp32/model_data.h`

## 5. Flash

1. Open `firmware/esp32/main.ino` in Arduino IDE
2. Select board: "ESP32S3 Dev Module"
3. Set PSRAM: "OPI PSRAM"
4. Upload speed: 921600
5. Click Upload

## 6. Serial monitor

Open Serial Monitor at 115200 baud.
You should see:
```
System ready. Listening...
[87ms] background (0.92)
[91ms] fire_alarm (0.88)   ← ALERT triggered
```

## Pin connections

| Component | ESP32-S3 pin |
|-----------|-------------|
| INMP441 SCK | GPIO 14 |
| INMP441 WS | GPIO 15 |
| INMP441 SD | GPIO 13 |
| WS2812B data | GPIO 5 |
| ERM motor (MOSFET gate) | GPIO 6 |
