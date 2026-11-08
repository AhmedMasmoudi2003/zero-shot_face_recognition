import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from app.core.config import settings

class FaceEmbedder:
    def __init__(self, image_size: int = settings.IMAGE_SIZE, margin: int = settings.MARGIN):
        """
        Initialize MTCNN for face detection/alignment and InceptionResnetV1 for embeddings.
        """
        self.device = torch.device(settings.DEVICE)
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
        Given a BGR frame, detect a face and return (embedding, bounding_box).
        Returns:
            embedding: np.ndarray of shape (512,) or None if no face.
            box: [x1, y1, x2, y2] or None.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        boxes, probs = self.mtcnn.detect(pil_img)

        if boxes is None or len(boxes) == 0:
            return None, None

        best_idx = int(np.argmax(probs))
        best_box = boxes[best_idx]

        x1, y1, x2, y2 = [int(v) for v in best_box]
        h, w, _ = frame_bgr.shape
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

        face_rgb = frame_rgb[y1:y2, x1:x2]
        if face_rgb.size == 0:
            return None, None

        face_pil = Image.fromarray(face_rgb)
        aligned = self.mtcnn(face_pil)
        if aligned is None:
            return None, None

        aligned = aligned.to(self.device).unsqueeze(0)
        with torch.no_grad():
            embedding = self.model(aligned)
        
        embedding = embedding[0].cpu().numpy()

        # L2-normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding, [x1, y1, x2, y2]

# Singleton instance
face_embedder = FaceEmbedder()
