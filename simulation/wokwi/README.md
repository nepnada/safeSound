# Wokwi ESP32-S3 Circuit Simulation

## What is Wokwi?

[Wokwi](https://wokwi.com) is a free online electronics simulator. You can build circuits with an ESP32-S3, connect LEDs, buzzers, buttons, and **run real Arduino code** — all in your browser. No hardware needed.

## How to run our simulation

### Step 1: Open Wokwi

Go to: **https://wokwi.com/projects/new/esp32-s3-devkitc-1**

### Step 2: Set up the circuit

**Option A (quick):** Copy the contents of `diagram.json` in this folder → click the circuit editor tab → paste to replace.

**Option B (manual):** Add these components from the parts panel:
- 1× ESP32-S3 DevKitC
- 1× NeoPixel Ring/Strip (set to 8 pixels)
- 1× Piezo Buzzer (simulates vibration motor)
- 1× Green LED + 220Ω resistor (status indicator)
- 1× Push Button (test trigger)

Wire them:
| Component | Pin | ESP32-S3 Pin |
|-----------|-----|-------------|
| NeoPixel DIN | Data | GPIO 5 |
| NeoPixel VDD | Power | 3V3 |
| NeoPixel VSS | Ground | GND |
| Buzzer + | Signal | GPIO 6 |
| Buzzer − | Ground | GND |
| LED (via 220Ω) | Anode | GPIO 7 |
| Button | Input | GPIO 4 |
| INMP441 SCK | (label only) | GPIO 14 |
| INMP441 WS | (label only) | GPIO 15 |
| INMP441 SD | (label only) | GPIO 13 |

### Step 3: Paste the code

Copy the contents of `sketch.ino` → paste into the code editor on Wokwi.

### Step 4: Add the library

In the Library Manager (book icon), search and add: **Adafruit NeoPixel**

### Step 5: Run!

Click the green **Play** button. You'll see:
- The status LED starts breathing (listening state)
- Open the Serial Monitor at the bottom
- **Press the red TEST button** to cycle through sound classes

Each button press simulates a different sound detection:
1. **Fire alarm** → RED LEDs + 5 vibration pulses (CRITICAL)
2. **Baby crying** → ORANGE LEDs + 3 vibration pulses
3. **Choking** → PURPLE LEDs + 5 vibration pulses (CRITICAL)
4. **Car horn** → YELLOW LEDs + 3 vibration pulses
5. **Doorbell** → BLUE LEDs + 3 vibration pulses
6. **Background** → LEDs OFF, no vibration

## What the Serial Monitor shows

```
╔══════════════════════════════════════════════════════╗
║  Sound Alert System — ESP32-S3 Wokwi Simulation    ║
╠══════════════════════════════════════════════════════╣
║  Model: DS-CNN INT8 | 36.4 KB | 18,630 params     ║
║  Classes: 6 | Smoothing: 3 frames                 ║
║  Press TEST button to cycle through sound classes  ║
╚══════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────┐
│ Scenario: FIRE_ALARM                              │
├──────────────────────────────────────────────────┤
│ Class         | Raw Prob | Smoothed | Threshold  │
│ FIRE_ALARM    |  0.920   |  0.640   |  0.75     │
│ BABY_CRY      |  0.020   |  0.173   |  0.80     │
│ ...                                              │
├──────────────────────────────────────────────────┤
│ Predicted: FIRE_ALARM  | Latency: 2ms           │
└──────────────────────────────────────────────────┘
  🚨 ALERT: FIRE_ALARM | LED: RED | Vibration: 5 pulses
```

## Limitations

- **INMP441 microphone** is not available in Wokwi → we simulate audio input with a button
- **TFLite Micro** cannot run in Wokwi → we simulate inference with pre-computed scores
- For **real model inference**, use `python simulation/embedded_simulator.py` instead

## Circuit schematic

See `notebooks/figures/circuit_schematic.png` for a professional wiring diagram.
