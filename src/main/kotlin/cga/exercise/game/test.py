# Install FastAPI and Uvicorn if you haven't
# pip install fastapi uvicorn
import time
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.get("/1")
async def endpoint1():
    print("testpy")
    time.sleep(5)
    print("bye")
# Define a request model
#class DataModel(BaseModel):
 #   id: int = 1
  #  message: str = "testpy"

# POST route to receive data from the client
#@app.post("/send/")
#async def receive_data(data: DataModel):
 #   return {"received_id": data.id, "received_message": data.message}

# GET route to send data to the client
#@app.get("/get/")
#async def send_data():
 #   return {"id": 1, "message": "Hello from FastAPI!"}

# Run the server using `uvicorn`:
#python -m uvicorn test:app --reload