from app.db.database import engine, Base
# Import all models to ensure they are registered with Base
from app.models.employee import Employee
from app.models.attendance import Attendance
from app.models.department import Department
from app.models.employee_image import EmployeeImage

def reset_db():
    print("Dropping all tables...")
    Base.metadata.drop_all(bind=engine)
    print("Creating all tables...")
    Base.metadata.create_all(bind=engine)
    print("Database reset complete.")

if __name__ == "__main__":
    reset_db()
