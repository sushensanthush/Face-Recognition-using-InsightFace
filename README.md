# InsightFace Face Comparison

A simple face verification tool built with InsightFace and OpenCV.

This project extracts facial embeddings from two images and compares them using:

- Cosine Similarity
- Euclidean Distance

The script automatically selects the largest detected face when multiple faces are present and provides a basic confidence estimate.

## Features

- Face detection with InsightFace (buffalo_l model)
- Face embedding extraction
- Cosine similarity calculation
- Euclidean distance calculation
- Multiple-face handling
- Error handling for missing images or undetected faces

## Requirements

```bash
pip install insightface opencv-python numpy onnxruntime-gpu
```

For CPU:

```bash
pip install insightface opencv-python numpy onnxruntime
```

## Usage

Place your images in the project folder:

```
person1.webp
person2.webp
```

Run:

```bash
python compare.py
```

Example output:

```text
==================================================
FACE COMPARISON
==================================================
Cosine Similarity : 0.8123
Euclidean Distance: 0.6231
Match Confidence : Very High
```

## How It Works

1. Detect faces in both images.
2. Select the largest detected face.
3. Extract face embeddings using InsightFace.
4. Compare embeddings using cosine similarity and Euclidean distance.
5. Return a confidence estimate.

## Model

This project uses the `buffalo_l` model provided by InsightFace.

## Disclaimer

This project is intended for educational and research purposes. Face recognition results are probabilistic and should not be used as the sole basis for identity verification.
