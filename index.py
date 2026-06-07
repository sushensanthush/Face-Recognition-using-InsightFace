import cv2
import numpy as np
from insightface.app import FaceAnalysis


def get_primary_face(face_analyzer, image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    faces = face_analyzer.get(img)

    if len(faces) == 0:
        raise ValueError(f"No face detected in {image_path}")

    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )

    return face


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)



fa = FaceAnalysis(name="buffalo_l")
fa.prepare(ctx_id=0, det_size=(640, 640))


face1 = get_primary_face(fa, "person1.webp")
face2 = get_primary_face(fa, "person2.webp")

emb1 = face1.embedding
emb2 = face2.embedding


distance = euclidean_distance(emb1, emb2)
similarity = cosine_similarity(emb1, emb2)

print("=" * 50)
print("Face Comparison Results")
print("=" * 50)
print(f"Euclidean Distance : {distance:.4f}")
print(f"Cosine Similarity  : {similarity:.4f}")


if similarity > 0.65:
    verdict = "LIKELY SAME PERSON"
elif similarity > 0.50:
    verdict = "UNCERTAIN"
else:
    verdict = "DIFFERENT PEOPLE"

print(f"Verdict            : {verdict}")