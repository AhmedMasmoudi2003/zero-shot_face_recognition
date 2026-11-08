from typing import List, Optional
from sqlalchemy.orm import Session
from app.repositories.base import BaseRepository
from app.models.attendance import Attendance
import datetime

class AttendanceRepository(BaseRepository[Attendance]):
    def get_by_employee_and_date(
        self, db: Session, employee_id: str, date: datetime.date
    ) -> List[Attendance]:
        # Debug: Print the parameters
        print(f"DEBUG: Searching attendance for employee_id={employee_id}, date={date}")
        
        # Convert employee_id to UUID if it's a string
        try:
            import uuid
            if isinstance(employee_id, str):
                employee_uuid = uuid.UUID(employee_id)
                print(f"DEBUG: Converted employee_id to UUID: {employee_uuid}")
            else:
                employee_uuid = employee_id
        except ValueError as e:
            print(f"DEBUG: Invalid UUID format: {e}")
            return []
        
        query = db.query(self.model).filter(
            self.model.employee_id == employee_uuid,
            self.model.date == date
        )
        
        # Debug: Print the SQL query
        print(f"DEBUG: SQL Query: {query}")
        
        results = query.all()
        print(f"DEBUG: Found {len(results)} records")
        
        return results

    def get_active_attendance(self, db: Session, employee_id: str, date: datetime.date) -> Optional[Attendance]:
        # Convert employee_id to UUID if it's a string
        try:
            import uuid
            if isinstance(employee_id, str):
                employee_uuid = uuid.UUID(employee_id)
            else:
                employee_uuid = employee_id
        except ValueError:
            return None
            
        return db.query(self.model).filter(
            self.model.employee_id == employee_uuid,
            self.model.date == date,
            self.model.check_out == None
        ).first()

attendance_repo = AttendanceRepository(Attendance)
