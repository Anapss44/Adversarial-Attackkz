# 🛡️ Adversarial Attack Lab

A full-stack adversarial machine learning application with **5 attack types** and an **ensemble detector**.

---

## Features

### ⚡ Attack Lab
| Attack | Type | Description |
|--------|------|-------------|
| **FGSM** | Gradient | Fast Gradient Sign Method — single-step, fastest |
| **PGD** | Iterative | Projected Gradient Descent — stronger iterative attack |
| **BIM** | Iterative | Basic Iterative Method (I-FGSM) — multi-step FGSM |
| **DeepFool** | Geometry | Finds minimal perturbation to cross decision boundary |
| **C&W** | Optimization | Carlini & Wagner L2 — strongest white-box attack |

### 🔍 Ensemble Detector
Classifies any uploaded image as **CLEAN / SUSPICIOUS / ADVERSARIAL** using two signals:

**Statistical Analysis (45% weight)**
- High-frequency energy (Laplacian)
- Gradient magnitude distribution
- Local patch variance statistics
- Pixel kurtosis (tail analysis)
- Cross-channel correlation

**Model-Based Signals (55% weight)**
- Input squeezing via median filter
- JPEG compression robustness test
- Prediction confidence drop analysis

### 📋 History & 📊 Stats
- Full SQLite logging of all attacks and detections
- Per-attack-type breakdown, success rates, top fooled classes
- Detection rates and ensemble score distributions

---

## Stack

```
Frontend  →  Vanilla HTML/CSS/JS (single file, zero deps)
Backend   →  Node.js + Express
Database  →  SQLite via sql.js (pure JS, no native build needed)
ML Engine →  Python + PyTorch (ResNet18 ImageNet)
```

---

## Setup

### 1. Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Node.js dependencies
```bash
npm install
```

### 3. Run
```bash
npm start
# Dev mode with auto-reload:
npm run dev
```

### 4. Open
```
http://localhost:3001
```

> **Note:** ResNet18 ImageNet weights (~45MB) are downloaded automatically on first run.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/attack` | Run attack on uploaded image |
| `POST` | `/api/detect` | Run ensemble detector on image |
| `GET` | `/api/history/attacks` | Fetch attack history |
| `GET` | `/api/history/detections` | Fetch detection history |
| `DELETE` | `/api/history/attacks/:id` | Delete attack record |
| `DELETE` | `/api/history/detections/:id` | Delete detection record |
| `GET` | `/api/stats` | Dashboard statistics |

### POST /api/attack
```json
// Form-data fields:
{
  "image": "<file>",
  "attack_type": "fgsm | pgd | bim | deepfool | cw",
  "params": "{\"epsilon\": 0.1, \"steps\": 40}"
}
```

### POST /api/detect
```json
// Form-data fields:
{ "image": "<file>" }
```

---

## Python Script Usage (standalone)

```bash
# Run an attack
python backend/adversarial.py attack <image> <attack_type> '<params_json>' <output_path>

# Run detector
python backend/adversarial.py detect <image>
```

Examples:
```bash
python backend/adversarial.py attack photo.jpg fgsm '{"epsilon": 0.1}' out.png
python backend/adversarial.py attack photo.jpg pgd '{"epsilon": 0.1, "alpha": 0.01, "steps": 40}' out.png
python backend/adversarial.py attack photo.jpg cw '{"c": 1.0, "steps": 100, "lr": 0.01}' out.png
python backend/adversarial.py detect photo.jpg
python backend/adversarial.py detect out.png
```

---

## Project Structure

```
adversarial-lab/
├── backend/
│   ├── server.js          # Express API (7 endpoints)
│   └── adversarial.py     # All 5 attacks + ensemble detector
├── frontend/
│   └── public/
│       └── index.html     # Full SPA (no framework)
├── db/                    # SQLite DB auto-created here
├── uploads/               # Uploaded images
│   └── adversarial/       # Generated adversarial images
├── requirements.txt       # Python deps
├── package.json
└── README.md
```

---

## Detector Thresholds (calibrated for ResNet18/ImageNet)

| Feature | Clean threshold | Suspicious | Adversarial |
|---------|----------------|------------|-------------|
| HF Energy | < 0.006 | 0.006–0.01 | > 0.01 |
| Gradient Mag | < 0.01 | 0.01–0.02 | > 0.02 |
| Conf Drop (Med) | < 8% | 8–15% | > 15% |
| Ensemble Score | < 0.40 | 0.40–0.60 | > 0.60 |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHON_CMD` | `python3` | Python executable (use `python` on Windows) |
| `PORT` | `3001` | Server port |
