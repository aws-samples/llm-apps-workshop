import sys
import subprocess
from mangum import Mangum
from fastapi import FastAPI
from app.api.api_v1.api import router as api_router

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "API for question answering bot"}

app.include_router(api_router, prefix="/api/v1")
handler = Mangum(app)
