from sqlalchemy.orm import Session
from app.repositories.department import department_repo
from app.models.department import Department
from typing import List, Optional

class DepartmentService:
    def create_department(self, db: Session, name: str, description: Optional[str] = None) -> Department:
        existing = department_repo.get_by_name(db, name)
        if existing:
            raise ValueError(f"Department with name '{name}' already exists.")
        return department_repo.create(db, {"name": name, "description": description})

    def get_departments(self, db: Session, skip: int = 0, limit: int = 100) -> List[Department]:
        return department_repo.get_multi(db, skip=skip, limit=limit)

    def get_department(self, db: Session, department_id: str) -> Optional[Department]:
        return department_repo.get(db, department_id)

    def delete_department(self, db: Session, department_id: str) -> Department:
        return department_repo.remove(db, id=department_id)

department_service = DepartmentService()
