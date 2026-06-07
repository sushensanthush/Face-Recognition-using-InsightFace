import logging
from typing import Optional
import cv2
from insightface.app import FaceAnalysis
import numpy as np


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def initialize_face_analyzer(model_name: str = "buffalo_l", ctx_id: int = 0, det_size: tuple = (640, 640)) -> FaceAnalysis:
    """Initializes and prepares the InsightFace FaceAnalysis application."""
    try:
        app = FaceAnalysis(name=model_name)
        app.prepare(ctx_id=ctx_id, det_size=det_size)
        return app
    except Exception as e:
        logging.error(f"Failed to initialize FaceAnalysis: {e}")
        raise

def get_face_embedding(app: FaceAnalysis, image_path: str) -> Optional[np.ndarray]:
    """Loads an image and extracts the embedding of the first detected face."""
    img = cv2.imread(image_path)
    if img is None:
        logging.error(f"Could not read image from path: {image_path}")
        return None

    faces = app.get(img)
    if not faces:
        logging.warning(f"No faces detected in image: {image_path}")
        return None


    return faces[0].embedding

def calculate_face_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculates the Euclidean distance between two face embeddings."""
    return float(np.linalg.norm(emb1 - emb2))

def main():
  
    img1_path = "person1.webp"
    img2_path = "person2.webp"

    
    logging.info("Initializing FaceAnalysis model...")
    fa = initialize_face_analyzer()


    logging.info("Extracting face embeddings...")
    embedding_a = get_face_embedding(fa, img1_path)
    embedding_b = get_face_embedding(fa, img2_path)

   
    if embedding_a is not None and embedding_b is not None:
        distance = calculate_face_distance(embedding_a, embedding_b)
        logging.info(f"Face comparison complete.")
        print(f"\n[RESULT] Euclidean Distance: {distance:.4f}")
        
       
        if distance < 1.2:
            print("[RESULT] Match Status: Same Person")
        else:
            print("[RESULT] Match Status: Different People")
    else:
        logging.error("Face verification failed due to missing embeddings.")

if __name__ == "__main__":
    main()