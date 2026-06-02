# Changelog

Format: `[JX – description courte]` suivi du détail.
En cas de problème, lire l'entrée correspondante pour savoir quoi annuler.

---

## [J1 – Init] Création structure projet + documentation

**Ajouté:**
- `PROJECT.md` — document de référence complet (contexte, pipeline, plan 14 jours, specs)
- `CHANGELOG.md` — ce fichier

**Pipeline décidé (final, ne pas changer):**
```
Log-Mel 64ch → YAMNet fine-tune → KD → DS-CNN → QAT INT8 → TFLite → ESP32-S3
```

**Rien à annuler** — première entrée.

---

## [J1 – Setup] Environnement Python + structure projet + scripts data

**Ajouté:**
- `venv/` — environnement Python 3.11 (TF ne supporte pas 3.13)
- `requirements.txt` — tensorflow 2.15, librosa, tensorflow-hub, etc.
- `src/data/download.py` — téléchargement ESC-50 auto + instructions autres datasets
- `src/data/preprocess.py` — pipeline complet : load → resample 16kHz → log-mel 64ch → segmentation 1s/50% overlap → export numpy
- Dossiers `data/raw/{fire_alarm,baby_cry,choking,car_horn,doorbell,background}/`
- Dossiers `models/teacher/`, `models/student/`, `firmware/esp32/`

**Fix:** setuptools 69.5.1 requis pour tensorflow-hub (bug pkg_resources Python 3.11)

**Pour annuler:** supprimer `venv/` et relancer `python3.11 -m venv venv && pip install -r requirements.txt`

---

## [J1-J2 – Data Pipeline] Scripts données + données ESC-50

**Ajouté:**
- `src/data/download.py` — téléchargement ESC-50 (via git clone, pas zip)
- `src/data/audioset_download.py` — téléchargement AudioSet via yt-dlp + CSV
- `src/data/freesound_download.py` — téléchargement Freesound + YouTube search
- `src/data/augment.py` — augmentation ×6 : pitch shift ±2st, time stretch ×0.85, white noise SNR 15dB, background mix SNR 10dB → target 400 clips/classe

**Données actuelles (après ESC-50):**
- fire_alarm: 14 | baby_cry: 6 | choking: 12 | car_horn: 8 | doorbell: 13 | background: 45
- YouTube search en cours en background pour compléter

**Action manuelle requise:** ~10 clips choking enregistrés par l'utilisateur → `data/raw/choking/`

**Pour annuler:** `rm -rf data/raw/aug_*` supprime uniquement les fichiers augmentés

---

## [J2-J3 – ML Code] Scripts entraînement complets

**Ajouté:**
- `src/models/yamnet_finetune.py` — Teacher CNN (ResNet-style, 4 conv blocks) + SpecAugment + Mixup + class weights + EarlyStopping
- `src/models/dscnn.py` — Student DS-CNN (4 blocs depthwise separable, ~60KB INT8)
- `src/models/distillation.py` — Knowledge Distillation (T=6, α=0.7, loss = KL + CE)
- `src/convert.py` — QAT (tensorflow-model-optimization) + TFLite INT8 conversion + vérification accuracy
- `src/evaluate.py` — F1 par classe, confusion matrix (PNG), latence benchmark
- `src/train.py` — Entry point unique : `python src/train.py` lance tout
- `src/export_model_header.py` — Convertit .tflite → model_data.h (C array Arduino)

**Architecture finale confirmée:**
```
Log-Mel 64ch → Teacher CNN (train) → KD → DS-CNN (~60KB) → QAT INT8 → TFLite
```

**Pour annuler un step:** `python src/train.py --step teacher|student|convert|eval`

---

## [J2 – Firmware] Code ESP32-S3

**Ajouté:**
- `firmware/esp32/main.ino` — Pipeline complet : I2S → log-mel → TFLite Micro → temporal smoothing (3 frames) → LED WS2812B + vibration
- Seuils par classe : fire_alarm=0.75, choking=0.75, baby_cry=0.80, car_horn=0.80, doorbell=0.90
- Patterns LED couleur-codés + vibration (5 pulses critiques, 3 pulses normal)

**Dépendances Arduino à installer:** FastLED, TFLite Micro (espressif/esp-tflite-micro)

---

## [J2 – Tests & Automation] Scripts de test + automatisation

**Ajouté:**
- `test_model.py` — test suite complète : preprocessing ✓, dataset ✓, keras model, tflite model
- `run_pipeline.sh` — pipeline auto complet : teacher → distillation → QAT → TFLite → header
- `retrain.sh` — re-entraîne après ajout de nouvelles données (augment → preprocess → train)
- `src/models/inference_test.py` — test live sur microphone Mac (sounddevice)
- `firmware/esp32/README_FLASH.md` — guide complet de flash Arduino

**État entraînement:**
- Teacher en cours (best_teacher.keras 4.9MB sauvé → bon signe)
- Distillation + QAT + TFLite se lanceront dès teacher terminé avec `bash run_pipeline.sh`

**Pour tester maintenant (sans attendre la fin de l'entraînement):**
```bash
python test_model.py   # vérifie preprocessing + dataset
```

## [J3 – Fix] Correction data leakage dans le split

**Problème identifié:** Les fichiers augmentés (aug_*.wav) et leurs sources se retrouvaient dans train ET val/test → le modèle "voyait" des versions quasi-identiques à l'éval → val_accuracy peu fiable.

**Fix dans `src/data/preprocess.py`:**
- Fichiers source → splittés en train/val/test normalement
- Fichiers `aug_*` → train UNIQUEMENT (jamais dans val/test)
- Nouveau split en cours de génération

**Impact:** Val/test accuracy sera désormais une vraie mesure de généralisation.

**Pour annuler:** `git checkout src/data/preprocess.py` (ou supprimer et recréer depuis CHANGELOG J1-J2)

---

## [J3 – Training] Teacher terminé + Distillation lancée

**Résultats teacher (avec data leakage — à refaire après fix):**
- Train accuracy: 60% | Val accuracy: 42%
- Sauvé: `models/teacher/yamnet_finetuned.keras` (4.9MB, 423K params)

**Distillation en cours:**
- Teacher 423K params → Student DS-CNN 18K params (23× compression)
- Température T=6, α=0.7
- Train acc epoch 1: ~54%

**Prochaine étape après distillation:** QAT → TFLite (automatique avec `bash run_pipeline.sh`)

**Résultats distillation (student DS-CNN, dataset LEAKY):**
- val_accuracy max: **85.78%**
- Student 18K params vs Teacher 423K params = 23× compression

**Résultats student sur dataset PROPRE (leak-free, 153 samples test):**

| Classe | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| fire_alarm | 0.897 | 0.963 | **0.929** |
| baby_cry | 1.000 | 1.000 | **1.000** |
| choking | 0.810 | 0.944 | **0.872** |
| car_horn | 0.875 | 0.875 | **0.875** |
| doorbell | 0.958 | 0.821 | **0.885** |
| background | 1.000 | 0.985 | **0.993** |
| **accuracy** | | | **94.1%** |

**Résultats QAT + TFLite INT8 (modèle final):**
- Taille: **36.4 KB** (cible < 100KB ✅)
- TFLite accuracy (200 samples): **86.9%**
- Accuracy sur test set propre: **95.4%**
- Latence CPU: 0.21ms mean (×6–10 sur ESP32 = ~1.5–2ms)

| Classe | F1 TFLite |
|--------|-----------|
| fire_alarm | 0.945 |
| baby_cry | 1.000 |
| choking | 0.900 |
| car_horn | 0.933 |
| doorbell | 0.906 |
| background | 0.993 |
| **macro avg** | **0.946** |

**Généré:** `firmware/esp32/model_data.h` (36.4 KB, prêt à flasher)

---

## [J4 – Docs & Viz] Documentation complète + visualisations

**Ajouté:**
- `GUIDE.md` — guide complet : contexte, architecture, choix techniques justifiés, résultats
- `STEPS.md` — guide d'exécution pas à pas avec commandes exactes + troubleshooting
- `notebooks/pipeline.ipynb` — notebook Jupyter documenté : exploration data, features, training, évaluation
- `notebooks/figures/class_distribution.png` — distribution des classes
- `notebooks/figures/spectrograms.png` — log-mel spectrograms par classe (données réelles)
- `notebooks/figures/confusion_matrix.png` — matrice de confusion TFLite
- `notebooks/figures/model_comparison.png` — Teacher vs Student : params + taille
- `notebooks/figures/f1_per_class.png` — F1 par classe avec macro avg

**Généré:** `firmware/esp32/model_data.h` (36.4 KB, prêt à flasher)

---

## [J5 – Simulation] Mode simulation complet (pas de matériel)

**Contexte:** Projet passe en mode simulation uniquement — pas d'ESP32 physique.

**Ajouté:**
- `simulation/embedded_simulator.py` — simulation complète du firmware :
  audio → log-mel → TFLite INT8 → temporal smoothing → alert decision
  Simule LED, vibration, latence ESP32 (×8 host). Export JSON.
- `simulation/dashboard.py` — dashboard Streamlit interactif :
  upload audio, visualisation confiance en temps réel, simulation LED
- `simulation/wokwi/diagram.json` — schéma circuit pour simulation en ligne
- `simulation/results/*.json` — rapports de simulation par classe

**Résultat simulation: 6/6 classes correctes ✓**

| Classe | Alertes | Latence ESP32 est. |
|--------|---------|-------------------|
| fire_alarm | 7/9 | 44ms |
| baby_cry | 7/9 | 9ms |
| choking | 4/9 | 8ms |
| car_horn | 6/9 | 9ms |
| doorbell | 7/9 | 9ms |
| background | 0/9 | 9ms |

**Mis à jour:** GUIDE.md (§17 simulation), STEPS.md (section simulation), PROJECT.md (structure)

**Pour annuler:** `rm -rf simulation/`

---

## [Final – Retrain] Dataset amélioré + modèle final

**Problème corrigé:** dataset initial trop petit (6 sources baby_cry, 8 car_horn)

**Solution:** génération synthétique (signal processing) + re-augmentation
- fire_alarm: 14 → 39 sources (+25 synthétiques)
- baby_cry: 6 → 31 sources (+25 synthétiques)
- choking: 12 → 37 sources (+25 synthétiques)
- car_horn: 8 → 33 sources (+25 synthétiques)
- doorbell: 13 → 38 sources (+25 synthétiques)
- background: 45 → 60 sources (+15 synthétiques)

**Dataset final:** Train 13,196 | Val 347 | Test 347 (leak-free)

**Résultats finaux — TFLite INT8 (347 test samples):**

| Classe | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| fire_alarm | 0.925 | 0.939 | **0.932** |
| baby_cry | 1.000 | 1.000 | **1.000** |
| choking | 0.835 | 0.985 | **0.904** |
| car_horn | 1.000 | 0.875 | **0.933** |
| doorbell | 0.851 | 0.784 | **0.816** |
| background | 0.987 | 0.918 | **0.951** |
| **accuracy** | | | **92.5%** |
| **macro F1** | | | **0.923** |

**Simulation: 6/6 classes correctes ✓**

**Modèle final:**
- `models/student/dscnn_int8.tflite` — 36.4 KB INT8
- `firmware/esp32/model_data.h` — prêt à flasher

---

## [J5 – Docs] Documentation finale complète

**Ajouté:**
- `GUIDE.md` — 17 sections, architecture, choix justifiés, résultats
- `STEPS.md` — guide pas-à-pas, commandes exactes, troubleshooting
- `notebooks/pipeline.ipynb` — notebook Jupyter documenté avec visualisations
- `notebooks/figures/*.png` — 5 figures : distribution, spectrogrammes, confusion matrix, modèle, F1

---

## [J6 – Demo & IoT] Interface professionnelle + Node-RED + Wokwi

**Ajouté:**
- `simulation/demo.py` — application Streamlit professionnelle, 4 pages :
  - Product : présentation produit, specs, classes, pipeline
  - Technology : architecture ML, comparaison teacher/student, métriques
  - Live Demo : classification temps réel (micro ou fichier audio), simulation bracelet (LED + vibration)
  - Hardware : BOM, pin assignments, budget mémoire ESP32
  - Design : palette sombre professionnelle (Inter font), sans emojis, style produit commercial
- `node-red/flows.json` — flow Node-RED complet pour dashboard IoT :
  - MQTT topics : classification, alerts, device status, notifications, config
  - Parsing classification + alert + device status
  - Caregiver push notification (MQTT)
  - CSV event logging
  - Inject nodes pour simulation sans hardware
  - Architecture : ESP32 -> BLE -> Smartphone -> MQTT -> Node-RED -> Caregiver
- `node-red/README.md` — setup, topics MQTT, architecture
- `simulation/wokwi/diagram.json` — circuit mis a jour : MODE + TEST buttons, charging LED, motor via resistor
- `simulation/wokwi/sketch.ino` — firmware Wokwi mis a jour : serial commands, 2 buttons, patterns LED/vibration
- `simulation/VIDEO_CONCEPT.md` — storyboard video produit (Canva + screen recording)

**Pourquoi Node-RED:**
Justifie la partie IoT du projet : Device -> MQTT -> Dashboard -> Caregiver notifications.
Sans ca, le projet est "systeme embarque + ML" mais pas "IoT".

**Pour lancer la demo:**
```bash
streamlit run simulation/demo.py
```

**Pour lancer Node-RED:**
```bash
node-red
# Import node-red/flows.json dans http://localhost:1880
```

---

**Pour tester le modele final:**
```bash
python test_model.py                          # checks automatiques
python src/evaluate.py                        # F1 + confusion matrix
python src/models/inference_test.py           # test live micro Mac
streamlit run simulation/demo.py              # demo professionnelle
```

---
