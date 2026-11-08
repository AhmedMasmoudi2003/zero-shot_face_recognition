from sqlalchemy import text
from app.db.database import engine

def fix_missing_columns():
    with engine.connect() as conn:
        print("Checking for missing columns...")
        try:
            # Check if department_id exists in employees
            conn.execute(text("SELECT department_id FROM employees LIMIT 1"))
            print("department_id already exists in employees table.")
        except Exception:
            print("Adding department_id to employees table...")
            conn.rollback() # Rollback the failed select
            with conn.begin():
                conn.execute(text("ALTER TABLE employees ADD COLUMN department_id UUID REFERENCES departments(id) ON DELETE SET NULL"))
            print("Column added successfully.")
            
        print("Ensuring all tables exist...")
        from app.models.employee import Employee
        from app.models.attendance import Attendance
        from app.models.department import Department
        from app.models.employee_image import EmployeeImage
        from app.db.database import Base
        Base.metadata.create_all(bind=engine)
        print("Schema synchronization complete.")

if __name__ == "__main__":
    fix_missing_columns()
