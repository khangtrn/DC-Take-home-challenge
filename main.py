from model import model_pipeline
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/ask")
def ask(userId: str):
    result = model_pipeline(userId)
    return result