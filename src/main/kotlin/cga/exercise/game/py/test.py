# Install FastAPI and Uvicorn if you haven't
# pip install fastapi uvicorn
#Ausführen in py ordner     python -m uvicorn test:app --reload
from fastapi import FastAPI
import random
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

def random_action():
    actions = ['W', 'A', 'S', 'D', 'P']
    return random.choice(actions)
# Endpoint to handle POST requests with GameData
@app.post("/send/")
async def receive_game_data(data: GameData):
    print("Received data:", data)

    # Simulate a response by sending back the received data with a slight modification
    response_data = GameData(
        spaceshipPosition=[pos for pos in data.spaceshipPosition],  # example modification
        spaceshipRotation=data.spaceshipRotation,
        asteroidPositions=data.asteroidPositions,
        action=random_action(),
        reward=data.reward,
        time=data.time
    )
    return response_data
