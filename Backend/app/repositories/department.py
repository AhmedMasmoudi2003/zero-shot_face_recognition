from app.repositories.base import BaseRepository
from app.models.department import Department
from sqlalchemy.orm import Session
from typing import Optional

class DepartmentRepository(BaseRepository[Department]):
    def get_by_name(self, db: Session, name: str) -> Optional[Department]:
        return db.query(self.model).filter(self.model.name == name).first()

department_repo = DepartmentRepository(Department)
