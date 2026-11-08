from fastapi import FastAPI
from app.api.routes import recognize, departments, employees
from app.core.config import settings
from app.db.database import engine, Base
# Import models to ensure they are registered with Base for create_all
from app.models.employee import Employee
from app.models.attendance import Attendance
from app.models.department import Department
from app.models.employee_image import EmployeeImage
from fastapi.staticfiles import StaticFiles
import os

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API for zero-shot face recognition using FaceNet and Pinecone"
)

# Mount static files for images
os.makedirs("static/images", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Wiring routes
app.include_router(recognize.router, tags=["Recognition & Attendance"])
app.include_router(departments.router, prefix="/departments", tags=["Departments"])
app.include_router(employees.router, prefix="/employees", tags=["Employees"])

@app.get("/")
async def root():
    return {"message": "Welcome to Zero-Shot Face Recognition API", "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
