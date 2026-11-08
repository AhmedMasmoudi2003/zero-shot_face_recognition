# face_embedder.py
import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from config import DEVICE


class FaceEmbedder:
    def __init__(self, image_size: int = 160, margin: int = 20):
        """
        Initialize MTCNN for face detection/alignment and InceptionResnetV1 for embeddings.
        """
        self.device = torch.device(DEVICE)
        # keep_all=False: we take only the main face
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=margin,
            keep_all=False,
            device=self.device,
            thresholds=[0.6, 0.7, 0.8]
        )
        self.model = InceptionResnetV1(pretrained="vggface2").to(self.device).eval()

    def frame_to_embedding(self, frame_bgr: np.ndarray):
        """
        Given a BGR frame from OpenCV, detect a face and return (embedding, bounding_box).

        Returns:
            embedding: np.ndarray of shape (512,) or None if no face.
            box: [x1, y1, x2, y2] or None.
        """
        # Convert BGR (OpenCV) -> RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # MTCNN forward: we want both face tensor and box
        # .detect returns boxes & probabilities; .__call__ can return aligned face(s)
        # Easiest: use mtcnn to get aligned face tensor and separate detect for boxes
        boxes, probs = self.mtcnn.detect(pil_img)

        if boxes is None or len(boxes) == 0:
            return None, None

        # Take the box with highest probability
        best_idx = int(np.argmax(probs))
        best_box = boxes[best_idx]

        # Now extract aligned face only for that best box
        # mtcnn can take "pil_img" and "save_path" or directly return a tensor; 
        # but to control which box is used, we crop manually then run mtcnn again with keep_all=False
        x1, y1, x2, y2 = [int(v) for v in best_box]
        h, w, _ = frame_bgr.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        face_rgb = frame_rgb[y1:y2, x1:x2]
        if face_rgb.size == 0:
            return None, None

        face_pil = Image.fromarray(face_rgb)
        aligned = self.mtcnn(face_pil)
        if aligned is None:
            return None, None

        # aligned: tensor of shape (3, image_size, image_size)
        aligned = aligned.to(self.device).unsqueeze(0)  # [1, 3, H, W]
        with torch.no_grad():
            embedding = self.model(aligned)
        # embedding shape: [1, 512]
        embedding = embedding[0].cpu().numpy()

        # L2-normalize (safety)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding, [x1, y1, x2, y2]
