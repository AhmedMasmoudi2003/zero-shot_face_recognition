from sqlalchemy import Column, String, DateTime, UUID, ForeignKey, Date
import uuid
import datetime
from app.db.database import Base

class Attendance(Base):
    __tablename__ = "attendance"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id = Column(UUID(as_uuid=True), ForeignKey("employees.id"), nullable=False)
    date = Column(Date, default=datetime.date.today)
    check_in = Column(DateTime, default=datetime.datetime.utcnow)
    check_out = Column(DateTime, nullable=True)
    status = Column(String) # e.g. "Present", "Late"
