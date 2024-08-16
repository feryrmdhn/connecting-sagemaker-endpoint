from fastapi import FastAPI
from app.services import predict

app = FastAPI()

app.include_router(predict.router)
