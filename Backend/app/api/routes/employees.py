from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
from app.db.database import get_db
from app.services.employee_service import employee_service
from pydantic import BaseModel
import uuid

router = APIRouter()

from datetime import datetime as dt
import datetime

class PresenceInfo(BaseModel):
    is_present: bool
    entry_time: Optional[dt] = None

class EmployeeOverviewResponse(BaseModel):
    id: uuid.UUID
    name: str
    employee_code: str
    department_id: Optional[uuid.UUID]
    is_active: bool
    image: Optional[str] = None
    presence: PresenceInfo

    class Config:
        from_attributes = True

class AttendanceResponse(BaseModel):
    id: uuid.UUID
    date: datetime.date
    check_in: dt
    check_out: Optional[dt] = None
    status: Optional[str] = None

    class Config:
        from_attributes = True

class EmployeeDetailResponse(BaseModel):
    id: uuid.UUID
    name: str
    employee_code: str
    department_id: Optional[uuid.UUID]
    is_active: bool
    today_attendance: List[AttendanceResponse]

    class Config:
        from_attributes = True

class EmployeeResponse(BaseModel):
    id: uuid.UUID
    name: str
    employee_code: str
    department_id: Optional[uuid.UUID]
    is_active: bool

    class Config:
        from_attributes = True

@router.post("/", response_model=EmployeeResponse)
async def create_employee(
    name: str = Form(...),
    department_id: Optional[str] = Form(None),
    image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        if department_id == "null" or department_id == "":
            department_id = None
        return await employee_service.create_employee(db, name, department_id, image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/", response_model=List[EmployeeOverviewResponse])
def get_employees(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return employee_service.get_employees_with_presence(db, skip=skip, limit=limit)

@router.get("/{employee_id}", response_model=EmployeeDetailResponse)
def get_employee(employee_id: str, db: Session = Depends(get_db)):
    emp = employee_service.get_employee_detail(db, employee_id)
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")
    return emp

@router.get("/attendance/{employee_id}/{date}", response_model=List[AttendanceResponse])
def get_attendance_by_date(employee_id: str, date: str, db: Session = Depends(get_db)):
    try:
        return employee_service.get_attendance_by_date(db, employee_id, date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format, use YYYY-MM-DD")

@router.delete("/{employee_id}")
def delete_employee(employee_id: str, db: Session = Depends(get_db)):
    emp = employee_service.delete_employee(db, employee_id)
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")
    return {"message": "Employee deleted"}

@router.post("/{employee_id}/images")
async def add_employee_image(employee_id: str, image: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        return await employee_service.add_image(db, employee_id, image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
