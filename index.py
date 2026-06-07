import cv2
import numpy as np
from insightface.app import FaceAnalysis


def get_face_embedding(face_analyzer, image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    faces = face_analyzer.get(img)

    if not faces:
        raise ValueError(f"No face detected in {image_path}")

   
    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )

    return face.embedding


def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )



fa = FaceAnalysis(name="buffalo_l")
fa.prepare(ctx_id=0, det_size=(640, 640))


emb1 = get_face_embedding(fa, "person1.webp")
emb2 = get_face_embedding(fa, "person2.webp")


euclidean_distance = np.linalg.norm(emb1 - emb2)
cosine_score = cosine_similarity(emb1, emb2)

print("=" * 50)
print("FACE COMPARISON")
print("=" * 50)
print(f"Cosine Similarity : {cosine_score:.4f}")
print(f"Euclidean Distance: {euclidean_distance:.4f}")


if cosine_score >= 0.75:
    print("Match Confidence : Very High")
elif cosine_score >= 0.65:
    print("Match Confidence : High")
elif cosine_score >= 0.55:
    print("Match Confidence : Medium")
else:
    print("Match Confidence : Low / Different Person")