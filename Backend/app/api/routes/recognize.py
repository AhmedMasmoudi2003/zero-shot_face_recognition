from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
import cv2
import numpy as np
from sqlalchemy.orm import Session
from app.services.face_embedder import face_embedder
from app.services.pinecone_client import pinecone_client
from app.services.attendance import attendance_service
from app.core.config import settings
from app.db.database import get_db

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "ok", "project": settings.PROJECT_NAME, "version": settings.VERSION}

@router.post("/recognize")
async def recognize_face(image: UploadFile = File(...)):
    # Read image
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Extract embedding
    embedding, box = face_embedder.frame_to_embedding(frame)
    if embedding is None:
        return {"match": False, "label": "No face detected"}

    # Query Pinecone
    result = pinecone_client.query_face(embedding.tolist())
    
    if result and hasattr(result, 'matches') and len(result.matches) > 0:
        top_match = result.matches[0]
        if top_match.score >= settings.SIMILARITY_THRESHOLD:
            name = top_match.metadata.get("name", top_match.id)
            return {
                "match": True,
                "person_id": top_match.id,
                "name": name,
                "score": top_match.score,
                "box": box
            }

    return {"match": False, "label": "Unknown", "score": result.matches[0].score if result.matches else 0}

@router.post("/attendance")
async def mark_attendance(image: UploadFile = File(...), db: Session = Depends(get_db)):
    # Recognizing first
    recognition = await recognize_face(image)
    if recognition.get("match"):
        # Log attendance
        log = attendance_service.log_attendance(db, recognition["person_id"], recognition["name"])
        return {"status": "success", "attendance": log}
    
    raise HTTPException(status_code=404, detail="Face not recognized, attendance not logged")
