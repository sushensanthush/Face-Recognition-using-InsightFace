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
    """Loads image, scales down safely, and returns (embedding, cropped_face, processed_image)."""
    img = cv2.imread(image_path)
    if img is None:
        logging.error(f"Could not read image: {image_path}")
        return None

    h, w, _ = img.shape
    MAX_DIMENSION = 1280
    if max(h, w) > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
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

def draw_rounded_panel(src: np.ndarray, top_left: Tuple[int, int], bottom_right: Tuple[int, int], color: Tuple[int, int, int], thickness: int, radius: int):
    """Draws a premium rounded panel border on the window canvas."""
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    cv2.line(src, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(src, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(src, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(src, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    
    cv2.ellipse(src, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(src, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(src, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
    cv2.ellipse(src, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)

def display_visual_results(crop1: np.ndarray, crop2: np.ndarray, similarity: float, threshold: float):
    """Generates a premium, high-fidelity UI layout for biometric comparison visualization."""
    canvas_w, canvas_h = 760, 480
    canvas = np.full((canvas_h, canvas_w, 3), 24, dtype=np.uint8)
    
    panel_size = (260, 260)
    face1_resized = cv2.resize(crop1, panel_size, interpolation=cv2.INTER_CUBIC)
    face2_resized = cv2.resize(crop2, panel_size, interpolation=cv2.INTER_CUBIC)
    
    y_offset = 150
    x_gap = 60
    x1_start, x1_end = x_gap, x_gap + panel_size[0]
    x2_start, x2_end = canvas_w - x_gap - panel_size[0], canvas_w - x_gap
    
    canvas[y_offset:y_offset+panel_size[1], x1_start:x1_end] = face1_resized
    canvas[y_offset:y_offset+panel_size[1], x2_start:x2_end] = face2_resized

    panel_color = (60, 60, 60)
    draw_rounded_panel(canvas, (x1_start-4, y_offset-4), (x1_end+4, y_offset+panel_size[1]+4), panel_color, 2, 12)
    draw_rounded_panel(canvas, (x2_start-4, y_offset-4), (x2_end+4, y_offset+panel_size[1]+4), panel_color, 2, 12)

    is_match = similarity >= threshold
    status_color = (120, 220, 100) if is_match else (90, 90, 235) 
    status_text = "VERIFIED MATCH" if is_match else "ACCESS DENIED"
    
    confidence_pct = max(0.0, min(100.0, (similarity + 1) * 50 if similarity < 0 else (similarity * 100)))

    
    cv2.rectangle(canvas, (0, 0), (canvas_w, 95), (34, 34, 34), -1)
    cv2.line(canvas, (0, 95), (canvas_w, 95), (50, 50, 50), 1)

    cv2.putText(canvas, "BIOMETRIC IDENTITY VERIFICATION", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Engine: InsightFace (Buffalo_L) | Threshold: >= {threshold:.2f}", (25, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    cv2.putText(canvas, status_text, (canvas_w - 240, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)

    cv2.putText(canvas, f"Cosine Score: {similarity:.4f}", (x1_start, y_offset + panel_size[1] + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Confidence: {confidence_pct:.1f}%", (x2_start + 60, y_offset + panel_size[1] + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imshow("Secure Face Verification Dashboard", canvas)
    logging.info("Professional visualization rendered successfully.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    img1_path = "person1.webp"
    img2_path = "person2.webp"
    
    COSINE_THRESHOLD = 0.45 

    logging.info("Initializing FaceAnalysis Core Architecture...")
    fa = initialize_face_analyzer()

    logging.info("Processing Target Face Matrix A...")
    face1_data = get_face_data(fa, img1_path)
    
    logging.info("Processing Target Face Matrix B...")
    face2_data = get_face_data(fa, img2_path)

    if face1_data and face2_data:
        emb1, crop1, _ = face1_data
        emb2, crop2, _ = face2_data

        similarity = calculate_cosine_similarity(emb1, emb2)
        
        print("\n" + "="*50)
        print(f" SYSTEM ANALYSIS COMPLETE")
        print(f" -> Cosine Vector Similarity : {similarity:.4f}")
        print(f" -> Configured Threshold    : {COSINE_THRESHOLD}")
        print("="*50 + "\n")
        
        display_visual_results(crop1, crop2, similarity, COSINE_THRESHOLD)
    else:
        logging.error("Execution terminated. One or both image inputs failed validation checks.")

if __name__ == "__main__":
    main()