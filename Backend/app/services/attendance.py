from sqlalchemy.orm import Session
from app.repositories.employee import employee_repo
from app.repositories.attendance import attendance_repo
from app.models.attendance import Attendance
import datetime

class AttendanceService:
    def log_attendance(self, db: Session, face_vector_id: str, name: str):
        # 1. Find the employee by their face vector ID
        employee = employee_repo.get_by_face_vector_id(db, face_vector_id)
        
        # If employee doesn't exist in our DB, return an error
        if not employee:
            return {
                "action": "error",
                "message": f"Face recognized as '{name}' but not registered as an employee in the company",
                "face_vector_id": face_vector_id,
                "status": "employee_not_found"
            }

        today = datetime.date.today()
        
        # 2. Check for an open attendance record for today
        active_attendance = attendance_repo.get_active_attendance(db, employee.id, today)

        if active_attendance:
            # 3. If there's an open record, mark as check-out
            updated_attendance = attendance_repo.update(db, db_obj=active_attendance, obj_in={
                "check_out": datetime.datetime.utcnow()
            })
            return {
                "action": "check_out",
                "id": str(updated_attendance.id),
                "employee_id": str(updated_attendance.employee_id),
                "date": str(updated_attendance.date),
                "check_in": updated_attendance.check_in.isoformat() if updated_attendance.check_in else None,
                "check_out": updated_attendance.check_out.isoformat() if updated_attendance.check_out else None,
                "status": updated_attendance.status
            }
        else:
            # 4. Otherwise, create a new check-in record
            new_attendance = attendance_repo.create(db, {
                "employee_id": employee.id,
                "date": today,
                "check_in": datetime.datetime.utcnow(),
                "status": "Present"
            })
            return {
                "action": "check_in",
                "id": str(new_attendance.id),
                "employee_id": str(new_attendance.employee_id),
                "date": str(new_attendance.date),
                "check_in": new_attendance.check_in.isoformat() if new_attendance.check_in else None,
                "check_out": None,
                "status": new_attendance.status
            }

# Singleton instance
attendance_service = AttendanceService()
