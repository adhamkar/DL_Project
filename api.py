# api.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Store the latest response
latest_response = None

class ResponseModel(BaseModel):
    response: str

@app.post("/set_response")
def set_response(response: ResponseModel):
    """
    Endpoint to set the latest response.
    """
    global latest_response
    latest_response = response.response
    return {"message": "Response set successfully."}

@app.get("/get_response")
def get_response():
    """
    Endpoint to get the latest response.
    """
    global latest_response
    if latest_response is None:
        return {"error": "No response available."}
    return {"response": latest_response}