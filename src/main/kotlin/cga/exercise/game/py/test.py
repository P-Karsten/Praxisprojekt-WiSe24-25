# Install FastAPI and Uvicorn if you haven't
# pip install fastapi uvicorn
import time
from fastapi import FastAPI

import time
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Define the GameData model
class Vector3f1(BaseModel):
    x: float
    y: float
    z: float

class GameData(BaseModel):
    spaceshipPosition: List[float]
    spaceshipRotation: Vector3f1
    asteroidPositions: List[List[float]]
    action: str
    reward: float
    time: float

# Endpoint to handle POST requests with GameData
@app.post("/send/")
async def receive_game_data(data: GameData):
    print("Received data:", data)

    # Simulate a response by sending back the received data with a slight modification
    response_data = GameData(
        spaceshipPosition=[pos for pos in data.spaceshipPosition],  # example modification
        spaceshipRotation=data.spaceshipRotation,
        asteroidPositions=data.asteroidPositions,
        action="processed_" + data.action,
        reward=data.reward,
        time=data.time
    )
    return response_data

# Endpoint to handle GET requests and return a simulated GameData object
@app.get("/get/")
async def send_game_data():
    # Simulate data that could be sent in a GET request
    simulated_data = GameData(
        spaceshipPosition=[1.0, 2.0, 3.0],
        spaceshipRotation=Vector3f1(x=0.0, y=1.0, z=0.0),
        asteroidPositions=[[5.5, 6.6, 7.7], [1.1, 2.2, 3.3]],
        action="simulate_action",
        reward=50.0,
        time=100.0
    )
    print("Sending simulated data:", simulated_data)
    return simulated_data
from pydantic import BaseModel

#app = FastAPI()
#aktuelle_zeit_in_ms = int(time.time() * 1000)
#print(aktuelle_zeit_in_ms)
#@app.get("/1")
#async def endpoint1():
#    aktuelle_zeit_in_ms = int(time.time() * 1000)
#    print(aktuelle_zeit_in_ms)
#    print("testpy")
#    time.sleep(5)
#    print("bye")
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