import logging
from typing import Optional, Tuple
import cv2
from insightface.app import FaceAnalysis
import numpy as np


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def initialize_face_analyzer(model_name: str = "buffalo_l", ctx_id: int = 0) -> FaceAnalysis:
    """Initializes FaceAnalysis model with a stable standard resolution."""
    try:
        app = FaceAnalysis(name=model_name)
       
        app.prepare(ctx_id=ctx_id, det_size=(960, 960))
        return app
    except Exception as e:
        logging.error(f"Failed to initialize FaceAnalysis: {e}")
        raise

def get_face_data(app: FaceAnalysis, image_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Loads image, scales down if resolution is too massive to prevent model failure, 
    and returns (embedding, cropped_face_image, processed_image).
    """
    img = cv2.imread(image_path)
    if img is None:
        logging.error(f"Could not read image: {image_path}")
        return None

    h, w, _ = img.shape
    

    MAX_DIMENSION = 1280
    if max(h, w) > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w, _ = img.shape
        logging.info(f"Scaled down massive image {image_path} to {w}x{h} for robust detection.")


    faces = app.get(img)
    if not faces:
        logging.warning(f"No faces detected in: {image_path}")
        return None


    faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
    primary_face = faces[0]


    bbox = primary_face.bbox.astype(int)
    x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(w, bbox[2]), min(h, bbox[3])
    cropped_face = img[y1:y2, x1:x2]

    return primary_face.embedding, cropped_face, img

def calculate_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculates Cosine Similarity between two raw face embeddings."""
    dot_prod = np.dot(emb1, emb2)
    norm_a = np.linalg.norm(emb1)
    norm_b = np.linalg.norm(emb2)
    
   
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_prod / (norm_a * norm_b))

def display_visual_results(crop1: np.ndarray, crop2: np.ndarray, similarity: float, threshold: float):
    """Resizes cropped faces and displays them side-by-side with comparison data."""
    size = (300, 300)
    crop1_resized = cv2.resize(crop1, size)
    crop2_resized = cv2.resize(crop2, size)

    combined_img = np.hstack((crop1_resized, crop2_resized))

    is_match = similarity >= threshold
    status_text = "MATCH: Same Person" if is_match else "MISMATCH: Different People"
    color = (0, 255, 0) if is_match else (0, 0, 255) 

    canvas = np.zeros((400, 600, 3), dtype=np.uint8)
    canvas[100:400, 0:600] = combined_img


    cv2.putText(canvas, f"Similarity: {similarity:.4f}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, f"Threshold: >= {threshold}", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(canvas, status_text, (330, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Face Verification Preview", canvas)
    logging.info("Visual preview window generated. Press any key on the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    img1_path = "person1.webp"
    img2_path = "person2.webp"
    
   
    COSINE_THRESHOLD = 0.45 

    logging.info("Initializing FaceAnalysis...")
    fa = initialize_face_analyzer()

    logging.info("Processing Face 1...")
    face1_data = get_face_data(fa, img1_path)
    
    logging.info("Processing Face 2...")
    face2_data = get_face_data(fa, img2_path)

    if face1_data and face2_data:
        emb1, crop1, _ = face1_data
        emb2, crop2, _ = face2_data

       
        similarity = calculate_cosine_similarity(emb1, emb2)
        
        print(f"\n[METRICS] Cosine Similarity: {similarity:.4f}")
        print(f"[METRICS] Decision Threshold: {COSINE_THRESHOLD}")
        
       
        display_visual_results(crop1, crop2, similarity, COSINE_THRESHOLD)
    else:
        logging.error("Verification aborted due to processing errors (Check if face is visible or resolution is too blurred).")

if __name__ == "__main__":
    main()