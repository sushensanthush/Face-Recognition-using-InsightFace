# Biometric Face Recognition

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python" />
  <img src="https://img.shields.io/badge/InsightFace-Buffalo_L-success?style=flat" />
  <img src="https://img.shields.io/badge/ONNX_Runtime-AI_Inference-orange?style=flat" />
  <img src="https://img.shields.io/badge/OpenCV-Computer_Vision-green?style=flat&logo=opencv" />
  <img src="https://img.shields.io/badge/NumPy-Vector_Math-blueviolet?style=flat&logo=numpy" />
  <img src="https://img.shields.io/badge/Tkinter-Desktop_UI-darkgreen?style=flat" />
  <img src="https://img.shields.io/badge/Pillow-Image_Processing-lightgrey?style=flat" />
  <img src="https://img.shields.io/badge/License-MIT-red?style=flat" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/AI_Powered-InsightFace-0A84FF?style=flat" />
  <img src="https://img.shields.io/badge/Enterprise-Ready-30D158?style=flat" />
  <img src="https://img.shields.io/badge/Biometric-Verification-FF9F0A?style=flat" />
  <img src="https://img.shields.io/badge/Cosine_Similarity-L2_Normalized-64D2FF?style=flat" />
  <img src="https://img.shields.io/badge/UI-Dark_Mode-1C1C1E?style=flat" />
</p>

<p align="center">
  <strong>Enterprise-Grade Facial Identity Verification Platform</strong><br/>
  Powered by InsightFace • ONNX Runtime • OpenCV • NumPy • Tkinter
</p>

##  Overview

The **Biometric Identity Verification Dashboard** is a premium desktop-based facial verification platform engineered for high-accuracy biometric identity comparison and enterprise verification workflows.

The platform combines a high-performance computer vision architecture with a modern Apple Dark Mode-inspired desktop experience, delivering fast, reliable, and visually refined facial verification capabilities.

At its core, the application leverages the **InsightFace Buffalo_L** deep learning model running on **ONNX Runtime**, enabling highly accurate facial landmark detection and high-dimensional embedding extraction. The verification engine performs mathematically precise identity comparisons using **L2-normalized vectors** and **Cosine Similarity scoring**, ensuring consistent results across varying image resolutions and input conditions.

---

#  Features

### Advanced AI Recognition Engine

- InsightFace Buffalo_L facial recognition model
- ONNX Runtime accelerated inference
- Facial landmark detection
- Deep embedding extraction
- Largest-face prioritization logic
- High-accuracy identity verification

### High-Performance Computer Vision

- OpenCV-powered image processing
- Safe image validation
- Real-time image downscaling
- Resolution optimization
- Memory-efficient processing pipeline

### Precision Matching System

- NumPy vectorized calculations
- Immediate L2 normalization
- Cosine Similarity verification
- Stable cross-resolution comparisons
- Enterprise-grade biometric scoring

### Premium User Experience

- Apple Dark Mode-inspired design
- Rounded image cards
- Anti-aliased rendering
- Dynamic verification banners
- Smooth UI interactions
- Professional enterprise aesthetic

### Smart File Workflow

- Drag-and-drop image uploads
- Native file browser support
- Live preview rendering
- Automatic image masking
- Interactive verification workspace

---

# Technology Stack

| Layer | Technology |
|---------|------------|
| AI Model | InsightFace Buffalo_L |
| Inference Engine | ONNX Runtime |
| Computer Vision | OpenCV (cv2) |
| Numerical Processing | NumPy |
| Desktop Framework | Tkinter |
| Drag & Drop | TkinterDnD2 |
| Image Rendering | Pillow (PIL) |
| Language | Python 3 |
| Logging | Python Logging |

---

# System Architecture

```text
┌─────────────────────────────┐
│ User Image Input            │
│ Drag & Drop / File Browser  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ OpenCV Processing Layer     │
│ Load • Validate • Resize    │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ InsightFace Buffalo_L       │
│ Face Detection & Alignment  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Facial Embedding Extraction │
│ Deep Feature Vector         │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ NumPy L2 Normalization      │
│ Vector Standardization      │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Cosine Similarity Engine    │
│ Identity Verification       │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Match / Mismatch Decision   │
└─────────────────────────────┘
```

---

# Verification Workflow

### Step 1 — Image Acquisition

Upload or drag two facial images into the verification workspace.

### Step 2 — Image Optimization

OpenCV validates and optimizes both images for analysis.

### Step 3 — Face Detection

InsightFace detects facial landmarks and aligns the facial structure.

### Step 4 — Embedding Generation

Deep facial feature vectors are extracted from each face.

### Step 5 — Vector Normalization

NumPy performs immediate L2 normalization.

### Step 6 — Similarity Calculation

Cosine Similarity is calculated between normalized embeddings.

### Step 7 — Verification Decision

A final identity verification decision is generated based on the configured threshold.

---

# Verification Mathematics

All embeddings are normalized before comparison:

```python
embedding = embedding / np.linalg.norm(embedding)
```

Cosine Similarity is calculated as:

```text
Similarity = A · B
```

Where:

- A = Normalized Face Embedding
- B = Normalized Face Embedding

Because both vectors are normalized:

```text
CosineSimilarity = A · B
```

This guarantees mathematically consistent biometric comparison regardless of image resolution.

---

# User Interface

## Premium Dashboard Layout

- Apple-inspired dark design language
- Enterprise control panel styling
- Rounded profile cards
- Anti-aliased image rendering
- Dynamic status indicators

## Verification States

| Status | Description |
|----------|------------|
| READY FOR VERIFICATION | Awaiting image input |
| COMPUTING VECTOR MATRICES | Verification processing |
| IDENTITY VERIFIED MATCH | Successful match |
| SECURITY NOTICE: MISMATCH | Identity mismatch |
| VERIFICATION INTERRUPTED | Face extraction failed |

---

# Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/biometric-verification-dashboard.git

cd biometric-verification-dashboard
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Requirements

```text
insightface
onnxruntime
opencv-python
numpy
Pillow
tkinterdnd2
```

---

# Run Application

```bash
python main.py
```

---

# Project Structure

```text
biometric-verification-dashboard/
│
├── assets/
│   ├── banner.png
│
├── main.py
├── requirements.txt
├── README.md
│
└── screenshots/
    ├── dashboard.png
    ├── verification-success.png
    └── verification-failed.png
```

---

# Performance Characteristics

| Component | Optimization |
|------------|-------------|
| Face Detection | InsightFace Buffalo_L |
| Inference | ONNX Runtime |
| Matrix Operations | NumPy Vectorization |
| Image Processing | OpenCV |
| Rendering | Pillow LANCZOS |
| UI Responsiveness | Tkinter Event Loop |

---

# Security & Accuracy

- Deep-learning facial embedding architecture
- Normalized vector comparison
- Deterministic similarity scoring
- Configurable verification threshold
- Enterprise-ready identity validation workflow

---

# Future Roadmap

- GPU CUDA acceleration
- Real-time webcam verification
- Face anti-spoofing detection
- Multi-face batch processing
- REST API integration
- Cloud deployment support
- Audit log generation
- LDAP / Active Directory integration

---

# Screenshots

### Main Dashboard

```text
screenshots/dashboard.png
```

### Verification Success

```text
screenshots/verification-success.png
```

### Verification Failure

```text
screenshots/verification-failed.png
```
---

Developed with:

- InsightFace
- ONNX Runtime
- OpenCV
- NumPy
- Pillow
- Tkinter

---

<p align="center">
  <strong>Biometric Identity Verification Dashboard</strong><br>
  Enterprise-Grade Facial Verification Platform
</p>
