# SafeSound Demo Video Concept

## Option 1: Canva Animated Video (Recommended)

Canva (free) has all the elements needed for a polished animated product demo.
No video editing skills required.

### Storyboard (6 scenes, ~60-90 seconds total)

**Scene 1 — The Problem (10s)**
- Dark background, white text
- Text: "In France, 6 million people live with hearing loss"
- Fade in: icons of fire alarm, car, doorbell
- Text: "Critical sounds go unheard. Every day."

**Scene 2 — Introducing SafeSound (8s)**
- Product name reveal: "SafeSound"
- Subtitle: "A wearable sound alert system"
- Animated bracelet illustration (simple flat design)

**Scene 3 — How It Works (15s)**
- Split screen:
  - Left: person wearing bracelet walking on a street (Canva stock illustration)
  - Right: 4-step pipeline animation
    1. Microphone icon -> "Captures ambient audio"
    2. Waveform -> spectrogram icon -> "Analyzes sound patterns"
    3. Brain/chip icon -> "ML classifies in < 2ms"
    4. Bracelet vibrating + LED -> "Alerts instantly"

**Scene 4 — Real-Life Scenario (20s)**
- Animated sequence (Canva illustrations):
  - Person walking on sidewalk wearing bracelet
  - Car approaches, honks (show sound wave animation)
  - Bracelet LEDs turn yellow, vibrates 3 times
  - Person looks up, steps aside
  - Text overlay: "Car horn detected - 91% confidence"
- Second scenario:
  - Person at home, smoke alarm goes off
  - Bracelet LEDs turn red, vibrates 5 rapid pulses
  - Text overlay: "Fire alarm detected - CRITICAL ALERT"

**Scene 5 — Technical Specs (10s)**
- Clean grid layout:
  - "36.4 KB model" | "95.4% accuracy"
  - "< 2ms latency" | "6 sound classes"
  - "Zero cloud dependency" | "8-12h battery"

**Scene 6 — IoT Integration (10s)**
- Bracelet -> phone -> caregiver notification
- "Family members receive real-time alerts remotely"
- Node-RED dashboard screenshot

**Closing (5s)**
- "SafeSound — Sound safety, on your wrist."
- Course info: "Projet de fin de module - Systemes Embarques IoT"

### How to Build in Canva

1. Go to canva.com -> Create -> Video (1920x1080)
2. Use template: search "product demo" or "tech presentation"
3. For bracelet illustration:
   - Search "smartwatch" or "wristband" in Elements
   - Or use the flat design template and add LED dots
4. For sound waves: search "sound wave" or "audio waveform" in Elements
5. Add animations: click element -> Animate -> choose subtle entrance
6. Export as MP4

### Audio for the video

- Background music: search "technology" or "innovation" in Canva audio
- No narration needed (text overlays are enough for a presentation)

---

## Option 2: Python Animation (manim or matplotlib)

If you want to generate the video programmatically:

```bash
pip install manim
```

Pros: fully reproducible, code-generated, impressive for a technical audience
Cons: steep learning curve, slower to produce

---

## Option 3: Screen Recording of the Demo App

Simplest approach — just screen-record the Streamlit demo:

1. Run `streamlit run simulation/demo.py`
2. Use QuickTime Player -> File -> New Screen Recording
3. Record: Product page -> Technology page -> Live Demo with audio
4. Add title/ending slides in Keynote or Canva
5. Export

This is the fastest option and shows the actual working system.

---

## Recommended Approach for Presentation

Combine Options 1 and 3:
- **Canva video** (scenes 1-2, 4, 6): the product story, real-life scenarios
- **Screen recording** (scenes 3, 5): the actual demo running
- Edit together in iMovie or Canva
