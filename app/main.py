from typing import List,Union
from uuid import uuid4
from fastapi import FastAPI
from models import User, Gender, Role

app = FastAPI()

db:List[User] = [
    User(
         id=uuid4(), 
         first_name="Jamila", 
         last_name="Ahmed",
         middle_name="Ahmed",
         gender=Gender.female,
         roles=[Role.student, Role.student]
    ),
    User(
         id=uuid4(), 
         first_name="Jamila2", 
         last_name="Ahmed2",
         middle_name="Ahmed",
         gender=Gender.female,
         roles=[Role.admin, Role.student]
    )
]

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/api/v1/users")
async def fetch_users():
    return db

@app.post("/api/v1/users")
async def register_user(user:User):
    db.append(user)
    return {"id": user.id}

