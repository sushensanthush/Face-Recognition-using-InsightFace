import logging
import os
from typing import Optional, Tuple
import cv2
from insightface.app import FaceAnalysis
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from tkinterdnd2 import DND_FILES, TkinterDnD


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FaceVerifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Biometric Identity Verification Dashboard")
        self.root.geometry("850x550")
        self.root.configure(bg="#181818") 
        
        
        self.img1_path: Optional[str] = None
        self.img2_path: Optional[str] = None
        self.fa: Optional[FaceAnalysis] = None
        self.COSINE_THRESHOLD = 0.45

     
        self.initialize_backend()
        
   
        self.build_ui()

    def initialize_backend(self):
        logging.info("Initializing FaceAnalysis Core Architecture...")
        try:
            self.fa = FaceAnalysis(name="buffalo_l")
            self.fa.prepare(ctx_id=0, det_size=(960, 960))
            logging.info("InsightFace loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize FaceAnalysis: {e}")
            messagebox.showerror("Backend Error", f"Could not load AI models:\n{e}")

    def build_ui(self):

        header_frame = tk.Frame(self.root, bg="#222222", height=80)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)

        title_label = tk.Label(
            header_frame, 
            text="BIOMETRIC IDENTITY VERIFICATION ENGINE", 
            font=("Segoe UI", 14, "bold"), 
            fg="#F0F0F0", 
            bg="#222222"
        )
        title_label.pack(anchor=tk.W, padx=25, pady=12)
        
        subtitle_label = tk.Label(
            header_frame, 
            text=f"Engine: InsightFace (Buffalo_L)  |  Decision Threshold: >= {self.COSINE_THRESHOLD}", 
            font=("Segoe UI", 9), 
            fg="#A0A0A0", 
            bg="#222222"
        )
        subtitle_label.place(x=25, y=45)

        
        self.workspace = tk.Frame(self.root, bg="#181818")
        self.workspace.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

      
        self.slot1 = self.create_drop_slot("TARGET FACE A (Drop or Click)", 0)
  
        self.slot2 = self.create_drop_slot("TARGET FACE B (Drop or Click)", 1)

        self.slot1.drop_target_register(DND_FILES)
        self.slot1.dnd_bind('<<Drop>>', lambda e: self.handle_file_drop(e, slot_index=0))
        self.slot2.drop_target_register(DND_FILES)
        self.slot2.dnd_bind('<<Drop>>', lambda e: self.handle_file_drop(e, slot_index=1))

        footer_frame = tk.Frame(self.root, bg="#181818", height=100)
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=30, pady=10)

        self.compare_btn = tk.Button(
            footer_frame, 
            text="RUN ANALYSIS", 
            font=("Segoe UI", 11, "bold"),
            bg="#383838", 
            fg="#FFFFFF", 
            activebackground="#505050", 
            activeforeground="#FFFFFF",
            bd=0, 
            padx=25, 
            pady=8,
            cursor="hand2",
            command=self.process_verification
        )
        self.compare_btn.pack(side=tk.RIGHT, pady=10)

       
        self.result_lbl = tk.Label(footer_frame, text="Status: Waiting for input vectors...", font=("Segoe UI", 12, "bold"), fg="#A0A0A0", bg="#181818")
        self.result_lbl.pack(side=tk.LEFT, anchor=tk.W, pady=5)
        
        self.metrics_lbl = tk.Label(footer_frame, text="", font=("Segoe UI", 10), fg="#808080", bg="#181818")
        self.metrics_lbl.place(x=0, y=35)

    def create_drop_slot(self, title: str, index: int) -> tk.Label:
        """Helper to create professional, rounded-looking image placeholders."""
        frame = tk.LabelFrame(self.workspace, text=f" {title} ", font=("Segoe UI", 9), bg="#181818", fg="#606060", bd=1, relief=tk.SOLID)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        display_label = tk.Label(frame, text="Drag & Drop Image Here\nor Click to Browse", font=("Segoe UI", 10), bg="#202020", fg="#808080", cursor="hand2")
        display_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
       
        display_label.bind("<Button-1>", lambda event: self.browse_file(index))
        return display_label

    def browse_file(self, slot_index: int):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.webp *.jpg *.jpeg *.png")])
        if file_path:
            self.load_image_to_slot(file_path, slot_index)

    def handle_file_drop(self, event, slot_index: int):
      
        file_path = event.data.strip('{}')
        if os.path.isfile(file_path) and file_path.lower().endswith(('.webp', '.jpg', '.jpeg', '.png')):
            self.load_image_to_slot(file_path, slot_index)
        else:
            messagebox.showwarning("File Format Error", "Please drop a valid image file (.webp, .jpg, .png)")

    def load_image_to_slot(self, path: str, slot_index: int):
        """Processes selected image file path and resizes thumbnails into the GUI panel frames."""
        if slot_index == 0:
            self.img1_path = path
            label_target = self.slot1
        else:
            self.img2_path = path
            label_target = self.slot2

        try:
          
            img = Image.open(path)
            img.thumbnail((280, 280))
            img_tk = ImageTk.PhotoImage(img)
            
            label_target.config(image=img_tk, text="")
            label_target.image = img_tk  
            logging.info(f"Loaded image into slot {slot_index}: {path}")
        except Exception as e:
            logging.error(f"UI display scale error: {e}")
            messagebox.showerror("UI Error", "Could not render image preview.")

    def get_face_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """Runs the image through InsightFace pipeline and extracts primary matrix vector."""
        img = cv2.imread(image_path)
        if img is None:
            return None

        
        h, w, _ = img.shape
        if max(h, w) > 1280:
            scale = 1280 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        faces = self.fa.get(img)
        if not faces:
            return None

      
        faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
        return faces[0].embedding

    def process_verification(self):
        """Main action handler triggering vector processing calculations."""
        if not self.img1_path or not self.img2_path:
            messagebox.showwarning("Input Missing", "Please select or drop both Image A and Image B profiles before running.")
            return

        self.result_lbl.config(text="Status: Processing vectors...", fg="#00A2FF")
        self.root.update_idletasks()

        emb1 = self.get_face_embedding(self.img1_path)
        emb2 = self.get_face_embedding(self.img2_path)

        if emb1 is None or emb2 is None:
            self.result_lbl.config(text="Status: Execution Failed", fg="#FF3B30")
            missing = []
            if emb1 is None: missing.append("Image A")
            if emb2 is None: missing.append("Image B")
            messagebox.showerror("Detection Failure", f"Could not map a distinct face profile in: {', '.join(missing)}")
            return

 
        dot_prod = np.dot(emb1, emb2)
        norm_a = np.linalg.norm(emb1)
        norm_b = np.linalg.norm(emb2)
        similarity = float(dot_prod / (norm_a * norm_b)) if (norm_a and norm_b) else 0.0
        confidence_pct = max(0.0, min(100.0, (similarity + 1) * 50 if similarity < 0 else (similarity * 100)))

      
        if similarity >= self.COSINE_THRESHOLD:
            self.result_lbl.config(text="VERIFIED MATCH", fg="#4CD964")
        else:
            self.result_lbl.config(text="ACCESS DENIED / MISMATCH", fg="#FF3B30") 

        self.metrics_lbl.config(
            text=f"Cosine Similarity Metrics Score: {similarity:.4f}  |  Calculated Match Confidence: {confidence_pct:.2f}%"
        )
        logging.info(f"Verification complete. Similarity Score: {similarity:.4f}")

if __name__ == "__main__":
    
    root = TkinterDnD.Tk()
    app = FaceVerifierApp(root)
    root.mainloop()