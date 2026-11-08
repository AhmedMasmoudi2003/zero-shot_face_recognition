from sqlalchemy import Column, String, UUID, Text
from sqlalchemy.orm import relationship
import uuid
from app.db.database import Base

class Department(Base):
    __tablename__ = "departments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)

    # Relationships
    employees = relationship("Employee", back_populates="department")
