# enroll.py
import cv2
import pinecone
from face_embedder import FaceEmbedder
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_ENVIRONMENT


def init_pinecone():
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    # Create index if not exists
    # For recent Pinecone, environment is set at project level; if needed, pass host or environment.
    if PINECONE_INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=512,
            metric="cosine"
        )
    index = pc.Index(PINECONE_INDEX_NAME)
    return index


def enroll_person():
    person_id = input("Enter unique person_id (e.g., student_001): ").strip()
    if not person_id:
        print("person_id cannot be empty")
        return

    display_name = input("Enter display name (e.g., Alice Smith): ").strip()
    if not display_name:
        display_name = person_id

    # Initialize
    index = init_pinecone()
    embedder = FaceEmbedder()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    print("Press SPACE to capture a frame for enrollment, or Q to quit.")
    captured_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        cv2.putText(frame, "Press SPACE to capture, Q to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("Enroll - Webcam", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):  # SPACE
            captured_frame = frame.copy()
            break
        elif key == ord("q") or key == 27:  # q or ESC
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured_frame is None:
        print("No frame captured; enrollment cancelled.")
        return

    # Extract embedding
    embedding, box = embedder.frame_to_embedding(captured_frame)
    if embedding is None:
        print("No face detected or multiple faces; please try again with a clear single face.")
        return

    # Upsert to Pinecone
    vector = {
        "id": person_id,
        "values": embedding.tolist(),
        "metadata": {
            "name": display_name
        }
    }
    index.upsert(vectors=[vector])

    print(f"Enrollment successful for {person_id} ({display_name}).")
    print("Vector dimension:", len(embedding))


if __name__ == "__main__":
    enroll_person()
