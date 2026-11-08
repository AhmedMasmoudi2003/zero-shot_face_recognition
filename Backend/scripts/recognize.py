# recognize.py
import time
import cv2
import numpy as np
import pinecone
import torch
import threading
from PIL import Image
from face_embedder import FaceEmbedder
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_ENVIRONMENT

SIMILARITY_THRESHOLD = 0.51  # tune this based on your tests
TOP_K = 1


def init_pinecone():
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
        raise RuntimeError(f"Index '{PINECONE_INDEX_NAME}' does not exist. "
                           f"Run enroll.py first to create it.")
    index = pc.Index(PINECONE_INDEX_NAME)
    return index


def main():
    index = init_pinecone()
    embedder = FaceEmbedder()

    print(f"Using device: {embedder.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    cap = cv2.VideoCapture(0)
    # Use higher resolution if possible, or keep 640x480 for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    print("--- Optimized Face Recognition ---")
    print("UI: Real-time (approx 30+ FPS)")
    print("Tracking: Real-time via MTCNN Thread")
    print("Recognition: Every 1.0s via Background Thread")
    print("Press Q to quit.")

    # Shared state for threads
    state = {
        "frame": None,      # Latest frame from camera
        "box": None,        # Current bounding box [x1, y1, x2, y2]
        "label": "No face", # Current identity label
        "color": (0, 0, 255), 
        "no_face_count": 0,
        "running": True
    }
    state_lock = threading.Lock()
    max_no_face_frames = 15 # Tolerance for tracking flicker

    def tracking_thread_func():
        """Continuously updates the bounding box."""
        while state["running"]:
            with state_lock:
                frame_to_track = state["frame"]
            
            if frame_to_track is None:
                time.sleep(0.01)
                continue
            
            # Detect faces (fast path)
            # Convert BGR -> RGB for MTCNN
            frame_rgb = cv2.cvtColor(frame_to_track, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # Use MTCNN's internal detect method for speed (no alignment/embedding here)
            boxes, probs = embedder.mtcnn.detect(pil_img)
            
            with state_lock:
                if boxes is not None and len(boxes) > 0:
                    best_idx = np.argmax(probs)
                    new_box = boxes[best_idx]
                    
                    if state["box"] is None:
                        state["box"] = new_box
                    else:
                        # Exponential smoothing to reduce "shaking" (alpha determines responsiveness)
                        # alpha=0.3 means 30% new detection, 70% previous position
                        alpha = 0.3
                        state["box"] = (new_box * alpha) + (state["box"] * (1 - alpha))
                    
                    state["no_face_count"] = 0
                else:
                    state["no_face_count"] += 1
                    if state["no_face_count"] > max_no_face_frames:
                        state["box"] = None
                        state["label"] = "No face"
                        state["color"] = (0, 0, 255)

    def recognition_thread_func():
        """Updates identity every 1.0 seconds."""
        last_query_time = 0
        query_interval = 1.0
        
        while state["running"]:
            now = time.time()
            if now - last_query_time < query_interval:
                time.sleep(0.1)
                continue
            
            with state_lock:
                frame_to_embed = state["frame"]
                if frame_to_embed is None or state["box"] is None:
                    continue
                # Take a copy to avoid mutation in tracking thread
                local_frame = frame_to_embed.copy()
            
            # Heavy lifting (Slow path)
            embedding, _ = embedder.frame_to_embedding(local_frame)
            
            if embedding is not None:
                # query Pinecone
                result = index.query(
                    vector=embedding.tolist(),
                    top_k=TOP_K,
                    include_metadata=True
                )

                new_label = "Unknown"
                new_color = (0, 0, 255)

                if result and hasattr(result, 'matches') and len(result.matches) > 0:
                    top_match = result.matches[0]
                    score = top_match.score
                    person_id = top_match.id
                    name = top_match.metadata.get("name", person_id) if hasattr(top_match, 'metadata') and top_match.metadata else person_id

                    if score >= SIMILARITY_THRESHOLD:
                        new_label = f"{name} ({score:.2f})"
                        new_color = (0, 255, 0)
                    else:
                        new_label = f"Unknown ({score:.2f})"
                
                with state_lock:
                    state["label"] = new_label
                    state["color"] = new_color
                    last_query_time = now
            else:
                last_query_time = now # Wait before trying again if embedding failed

    # Start background threads
    t_tracker = threading.Thread(target=tracking_thread_func, daemon=True)
    t_recognizer = threading.Thread(target=recognition_thread_func, daemon=True)
    t_tracker.start()
    t_recognizer.start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Update shared frame for background threads
            with state_lock:
                state["frame"] = frame.copy()
                curr_box = state["box"]
                curr_label = state["label"]
                curr_color = state["color"]

            # Draw results (UI Thread)
            if curr_box is not None:
                x1, y1, x2, y2 = [int(v) for v in curr_box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), curr_color, 2)
                
            # Draw label
            cv2.putText(frame, curr_label,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        curr_color, 2, cv2.LINE_AA)

            # Show visual indicator if recognition is "stale" (optional refinement)
            # For now, just show the frame
            cv2.imshow("Smart Face Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        state["running"] = False
        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
