from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.db.database import get_db
from app.services.department import department_service
from pydantic import BaseModel
import uuid

router = APIRouter()

class DepartmentCreate(BaseModel):
    name: str
    description: str | None = None

class DepartmentResponse(BaseModel):
    id: uuid.UUID
    name: str
    description: str | None = None
    
    class Config:
        from_attributes = True

@router.post("/", response_model=DepartmentResponse)
def create_department(dept: DepartmentCreate, db: Session = Depends(get_db)):
    try:
        return department_service.create_department(db, name=dept.name, description=dept.description)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/", response_model=List[DepartmentResponse])
def get_departments(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return department_service.get_departments(db, skip=skip, limit=limit)

@router.get("/{dept_id}", response_model=DepartmentResponse)
def get_department(dept_id: str, db: Session = Depends(get_db)):
    dept = department_service.get_department(db, dept_id)
    if not dept:
        raise HTTPException(status_code=404, detail="Department not found")
    return dept

@router.delete("/{dept_id}")
def delete_department(dept_id: str, db: Session = Depends(get_db)):
    dept = department_service.delete_department(db, dept_id)
    if not dept:
        raise HTTPException(status_code=404, detail="Department not found")
    return {"message": "Department deleted"}
