from fastapi import FastAPI
from .routers import toolcalling_service #, login, users, posts, votes
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(toolcalling_service.router)
# app.include_router(login.router)
# app.include_router(users.router)


@app.get("/")
def read_root():
    return {"mlops_mini": "This is a small end-to-end AI project for you!"}

