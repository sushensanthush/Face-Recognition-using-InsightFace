import logging
import os
import warnings
from typing import Optional, Tuple
import cv2
from insightface.app import FaceAnalysis
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
from tkinterdnd2 import DND_FILES, TkinterDnD

# Ignore layout future warnings from scikit-image to keep terminal clean
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FaceVerifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Biometric Identity Verification Dashboard")
        self.root.geometry("900x600")
        self.root.configure(bg="#121212") 
        self.root.resizable(False, False)
        
        self.img1_path: Optional[str] = None
        self.img2_path: Optional[str] = None
        self.fa: Optional[FaceAnalysis] = None
        self.COSINE_THRESHOLD = 0.45

        self.initialize_backend()
        self.build_ui()

    def initialize_backend(self):
        logging.info("Initializing FaceAnalysis Architecture...")
        try:
            # Force CPU execution explicitly to bypass ONNX provider warnings if needed
            self.fa = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
            self.fa.prepare(ctx_id=0, det_size=(640, 640)) 
            logging.info("InsightFace backend optimized successfully.")
        except Exception as e:
            logging.error(f"Backend architecture error: {e}")
            messagebox.showerror("Model Error", f"Failed to mount engine components:\n{e}")

    def create_rounded_rect(self, width: int, height: int, radius: int, bg_color: str) -> Image.Image:
        """Helper matrix to generate smooth anti-aliased card backgrounds."""
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle((0, 0, width, height), radius=radius, fill=bg_color)
        return img

    def build_ui(self):
        # 1. Top Navigation Banner Panel
        header_frame = tk.Frame(self.root, bg="#1E1E1E", height=90)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)

        border_line = tk.Frame(self.root, bg="#2C2C2C", height=1)
        border_line.pack(fill=tk.X, side=tk.TOP)

        title_label = tk.Label(
            header_frame, 
            text="BIOMETRIC IDENTITY ENGINE", 
            font=("Segoe UI", 15, "bold"), 
            fg="#F5F5F5", 
            bg="#1E1E1E"
        )
        title_label.pack(anchor=tk.W, padx=30, pady=15)
        
        subtitle_label = tk.Label(
            header_frame, 
            text=f"Core: Buffalo_L Matrix Matcher  |  System Threshold: >= {self.COSINE_THRESHOLD}", 
            font=("Segoe UI", 9), 
            fg="#8E8E93", 
            bg="#1E1E1E"
        )
        subtitle_label.place(x=30, y=52)

        # 2. Main Workspace Display Grid
        self.workspace = tk.Frame(self.root, bg="#121212")
        self.workspace.pack(fill=tk.BOTH, expand=True, padx=30, pady=25)

        self.canvas1, self.bg_img1 = self.create_drop_slot("VECTOR MATRIX SOURCE A", 0)
        self.canvas2, self.bg_img2 = self.create_drop_slot("VECTOR MATRIX SOURCE B", 1)

        # 3. Analytics Control Footer Frame
        footer_frame = tk.Frame(self.root, bg="#1E1E1E", height=110)
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM)
        footer_frame.pack_propagate(False)
        
        footer_border = tk.Frame(self.root, bg="#2C2C2C", height=1)
        footer_border.place(x=0, y=489)

        # Modern Action Button
        self.compare_btn = tk.Button(
            footer_frame, 
            text="VERIFY IDENTITY", 
            font=("Segoe UI", 11, "bold"),
            bg="#007AFF", 
            fg="#FFFFFF", 
            activebackground="#1485FF", 
            activeforeground="#FFFFFF",
            bd=0, 
            padx=30, 
            pady=10,
            cursor="hand2",
            command=self.process_verification
        )
        self.compare_btn.pack(side=tk.RIGHT, padx=30, pady=32)

        # Content Metrics Visual Panel
        self.result_lbl = tk.Label(footer_frame, text="READY FOR VERIFICATION", font=("Segoe UI", 13, "bold"), fg="#8E8E93", bg="#1E1E1E")
        self.result_lbl.pack(side=tk.LEFT, anchor=tk.W, padx=30, pady=25)
        
        # FIXED: Font size changed from 9.5 to 9 integer to prevent TclError
        self.metrics_lbl = tk.Label(footer_frame, text="Please input system data metrics to compute identity match scores.", font=("Segoe UI", 9), fg="#636366", bg="#1E1E1E")
        self.metrics_lbl.place(x=30, y=60)

    def create_drop_slot(self, title: str, index: int) -> Tuple[tk.Canvas, ImageTk.PhotoImage]:
        """Creates a modern UI card with responsive canvas rendering."""
        frame = tk.Frame(self.workspace, bg="#121212")
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20)

        title_lbl = tk.Label(frame, text=title, font=("Segoe UI", 9, "bold"), fg="#636366", bg="#121212")
        title_lbl.pack(anchor=tk.W, pady=5)

        card_w, card_h = 360, 280
        canvas = tk.Canvas(frame, width=card_w, height=card_h, bg="#121212", bd=0, highlightthickness=0, cursor="hand2")
        canvas.pack(fill=tk.BOTH, expand=True)

        card_base = self.create_rounded_rect(card_w, card_h, 16, "#1C1C1E")
        bg_img = ImageTk.PhotoImage(card_base)
        canvas.create_image(0, 0, anchor=tk.NW, image=bg_img)

        canvas.create_text(card_w//2, card_h//2 - 15, text="Drag & Drop Profile Here", font=("Segoe UI", 11, "bold"), fill="#AEAEB2", tags="placeholder")
        canvas.create_text(card_w//2, card_h//2 + 15, text="or click to browse filesystem", font=("Segoe UI", 9), fill="#636366", tags="placeholder")

        canvas.bind("<Button-1>", lambda event: self.browse_file(index))
        canvas.drop_target_register(DND_FILES)
        canvas.dnd_bind('<<Drop>>', lambda e: self.handle_file_drop(e, slot_index=index))

        return canvas, bg_img

    def browse_file(self, slot_index: int):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.webp *.jpg *.jpeg *.png")])
        if file_path:
            self.load_image_to_slot(file_path, slot_index)

    def handle_file_drop(self, event, slot_index: int):
        file_path = event.data.strip('{}')
        if os.path.isfile(file_path) and file_path.lower().endswith(('.webp', '.jpg', '.jpeg', '.png')):
            self.load_image_to_slot(file_path, slot_index)
        else:
            messagebox.showwarning("Data Format Error", "Supported file extensions: .webp, .jpg, .png")

    def load_image_to_slot(self, path: str, slot_index: int):
        if slot_index == 0:
            self.img1_path = path
            canvas_target = self.canvas1
        else:
            self.img2_path = path
            canvas_target = self.canvas2

        try:
            card_w, card_h = 360, 280
            src_img = Image.open(path)
            
            src_ratio = src_img.width / src_img.height
            card_ratio = card_w / card_h
            
            if src_ratio > card_ratio:
                new_h = card_h
                new_w = int(new_h * src_ratio)
                resized = src_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                left = (new_w - card_w) // 2
                cropped = resized.crop((left, 0, left + card_w, card_h))
            else:
                new_w = card_w
                new_h = int(new_w / src_ratio)
                resized = src_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                top = (new_h - card_h) // 2
                cropped = resized.crop((0, top, card_w, top + card_h))

            mask = Image.new("L", (card_w, card_h), 0)
            draw_mask = ImageDraw.Draw(mask)
            draw_mask.rounded_rectangle((0, 0, card_w, card_h), radius=16, fill=255)
            
            final_card = Image.new("RGBA", (card_w, card_h), (18, 18, 18, 0))
            final_card.paste(cropped, (0, 0), mask=mask)
            
            img_tk = ImageTk.PhotoImage(final_card)
            
            canvas_target.delete("all")
            canvas_target.create_image(0, 0, anchor=tk.NW, image=img_tk)
            
            if slot_index == 0: self.preview_img1 = img_tk
            else: self.preview_img2 = img_tk
            
            logging.info(f"Buffered matrix target input into channel index {slot_index}")
        except Exception as e:
            logging.error(f"Render pipeline breakdown: {e}")
            messagebox.showerror("UI Failure", "Processing render breakdown occurred.")

    def get_face_embedding(self, image_path: str) -> Optional[np.ndarray]:
        img = cv2.imread(image_path)
        if img is None: return None

        h, w, _ = img.shape
        if max(h, w) > 1080:
            scale = 1080 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        faces = self.fa.get(img)
        if not faces: return None

        faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
        raw_embedding = faces[0].embedding
        
        # L2 Normalization for highest comparison metric accuracy
        norm = np.linalg.norm(raw_embedding)
        if norm == 0: return raw_embedding
        return raw_embedding / norm

    def process_verification(self):
        if not self.img1_path or not self.img2_path:
            messagebox.showwarning("Data Flow Exception", "Both processing pipelines require target structural payloads.")
            return

        self.result_lbl.config(text="COMPUTING VECTOR MATRICES...", fg="#007AFF")
        self.root.update_idletasks()

        emb1 = self.get_face_embedding(self.img1_path)
        emb2 = self.get_face_embedding(self.img2_path)

        if emb1 is None or emb2 is None:
            self.result_lbl.config(text="VERIFICATION INTERRUPTED", fg="#FF453A")
            err_slots = [s for s, e in [("A", emb1), ("B", emb2)] if e is None]
            messagebox.showerror("Matrix Mapping Error", f"Biometric node point feature loss on Channel Profile: {', '.join(err_slots)}")
            return

        similarity = float(np.dot(emb1, emb2))
        confidence_pct = max(0.0, min(100.0, (similarity + 1) * 50 if similarity < 0 else (similarity * 100)))

        if similarity >= self.COSINE_THRESHOLD:
            self.result_lbl.config(text="IDENTITY VERIFIED MATCH", fg="#30D158") 
        else:
            self.result_lbl.config(text="SECURITY NOTICE: MISMATCH", fg="#FF453A") 

        self.metrics_lbl.config(
            text=f"Normalized Cosine Coefficient: {similarity:.4f}   |   Calculated Match Probability: {confidence_pct:.2f}%"
        )
        logging.info(f"Process verification loop exited with similarity output: {similarity:.4f}")

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = FaceVerifierApp(root)
    root.mainloop()