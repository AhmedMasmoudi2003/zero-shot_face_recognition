import os
import shutil
from typing import List, Optional
from sqlalchemy.orm import Session
from fastapi import UploadFile
from app.repositories.employee import employee_repo
from app.models.employee import Employee
from app.models.employee_image import EmployeeImage
from app.models.attendance import Attendance
from app.services.pinecone_client import pinecone_client
from app.services.face_embedder import face_embedder
import cv2
import numpy as np
import uuid

# Directory to store images
IMAGE_DIR = "static/images"
os.makedirs(IMAGE_DIR, exist_ok=True)

from app.repositories.attendance import attendance_repo
import datetime

class EmployeeService:
    def get_employees(self, db: Session, skip: int = 0, limit: int = 100) -> List[Employee]:
        return employee_repo.get_multi(db, skip=skip, limit=limit)

    def get_employees_with_presence(self, db: Session, skip: int = 0, limit: int = 100) -> List[dict]:
        employees = employee_repo.get_multi(db, skip=skip, limit=limit)
        today = datetime.date.today()
        result = []
        for emp in employees:
            active = attendance_repo.get_active_attendance(db, emp.id, today)
            first_image = emp.images[0].image_path if emp.images else None
            result.append({
                "id": emp.id,
                "name": emp.name,
                "employee_code": emp.employee_code,
                "department_id": emp.department_id,
                "is_active": emp.is_active,
                "image": first_image,
                "presence": {
                    "is_present": active is not None,
                    "entry_time": active.check_in if active else None
                }
            })
        return result

    def get_employee_detail(self, db: Session, employee_id: str) -> Optional[dict]:
        emp = employee_repo.get(db, employee_id)
        if not emp:
            return None
        today = datetime.date.today()
        history = attendance_repo.get_by_employee_and_date(db, str(emp.id), today)
        return {
            "id": emp.id,
            "name": emp.name,
            "employee_code": emp.employee_code,
            "department_id": emp.department_id,
            "is_active": emp.is_active,
            "today_attendance": history
        }

    def get_attendance_by_date(self, db: Session, employee_id: str, date_str: str) -> List[Attendance]:
        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        return attendance_repo.get_by_employee_and_date(db, employee_id, date_obj)

    def get_employee(self, db: Session, employee_id: str) -> Optional[Employee]:
        return employee_repo.get(db, employee_id)

    async def create_employee(
        self,
        db: Session,
        name: str,
        department_id: Optional[str],
        image: UploadFile
    ) -> Employee:
        # 1. Process Image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Invalid image")

        # 2. Get Embedding
        embedding, box = face_embedder.frame_to_embedding(frame)
        if embedding is None:
            raise ValueError("No face detected in image")

        # 3. Create Vector ID and Upsert to Pinecone
        face_vector_id = str(uuid.uuid4())
        pinecone_client.upsert_face(face_vector_id, embedding.tolist(), name)
        
        # 4. Create Employee in DB
        employee_code = f"EMP-{face_vector_id[:8].upper()}"
        employee = employee_repo.create(db, {
            "name": name,
            "employee_code": employee_code,
            "face_vector_id": face_vector_id,
            "department_id": department_id
        })

        # 5. Save Image to Disk
        img_filename = f"{employee.id}_{uuid.uuid4()}.jpg"
        img_path = os.path.join(IMAGE_DIR, img_filename)
        cv2.imwrite(img_path, frame)

        # 6. Create Image Record
        new_image = EmployeeImage(employee_id=employee.id, image_path=img_path)
        db.add(new_image)
        db.commit()
        
        return employee

    def update_employee(self, db: Session, employee_id: str, updates: dict) -> Employee:
        employee = employee_repo.get(db, employee_id)
        if not employee:
            raise ValueError("Employee not found")
        return employee_repo.update(db, db_obj=employee, obj_in=updates)

    def delete_employee(self, db: Session, employee_id: str):
        employee = employee_repo.get(db, employee_id)
        if not employee:
            return None
            
        # 1. Delete from Pinecone
        pinecone_client.delete_vector(employee.face_vector_id)
        
        # 2. Delete Images from Disk
        for img in employee.images:
            if os.path.exists(img.image_path):
                os.remove(img.image_path)

        # 3. Delete from DB
        return employee_repo.remove(db, id=employee_id)

    async def add_image(self, db: Session, employee_id: str, image: UploadFile):
        employee = employee_repo.get(db, employee_id)
        if not employee:
            raise ValueError("Employee not found")

        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
             raise ValueError("Invalid image")

        img_filename = f"{employee.id}_{uuid.uuid4()}.jpg"
        img_path = os.path.join(IMAGE_DIR, img_filename)
        cv2.imwrite(img_path, frame)

        new_image = EmployeeImage(employee_id=employee.id, image_path=img_path)
        db.add(new_image)
        db.commit()
        return new_image

employee_service = EmployeeService()
