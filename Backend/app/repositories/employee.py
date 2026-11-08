from typing import Optional
from sqlalchemy.orm import Session
from app.repositories.base import BaseRepository
from app.models.employee import Employee

class EmployeeRepository(BaseRepository[Employee]):
    def get_by_face_vector_id(self, db: Session, face_vector_id: str) -> Optional[Employee]:
        return db.query(self.model).filter(self.model.face_vector_id == face_vector_id).first()

employee_repo = EmployeeRepository(Employee)
