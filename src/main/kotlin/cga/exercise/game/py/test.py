# Install FastAPI and Uvicorn if you haven't
# pip install fastapi uvicorn
#Ausführen in py ordner     python -m uvicorn test:app --reload
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
import enum as Enum
from fastapi import Body
from starlette.responses import JSONResponse

app = FastAPI()


class Vector3f1(BaseModel):
    x: float
    y: float
    z: float

class GameData(BaseModel):
    spaceshipPosition: List[float]
    spaceshipRotation: Vector3f1
    closestAsteroid: Vector3f1

class Action(BaseModel):
    action: int =6

savedData: GameData = None


# Endpoint to handle POST requests with GameDatas
@app.post("/send/")
async def receive_game_data(data: GameData):
    response_data = GameData(
        spaceshipPosition=[pos for pos in data.spaceshipPosition],
        spaceshipRotation=data.spaceshipRotation,
        closestAsteroid=data.closestAsteroid,
        #reward=data.reward,
        #time=data.time
    )
    global savedData
    savedData = response_data
    return response_data


@app.get("/get")
async def send_game_data():
    if savedData is None:
        return {'Error: No data received'}

    print({
            "spaceshipPosition": savedData.spaceshipPosition,
            "spaceshipRotation": savedData.spaceshipRotation.y,
            "closestAsteroid": [
                savedData.closestAsteroid.x,
                savedData.closestAsteroid.y,
                savedData.closestAsteroid.z
            ]
        })

    return {
            "spaceshipPosition": savedData.spaceshipPosition,
            "spaceshipRotation": savedData.spaceshipRotation.y,
            "closestAsteroid": [
                savedData.closestAsteroid.x,
                savedData.closestAsteroid.y,
                savedData.closestAsteroid.z
            ]
        }

@app.post("/sendAction")
async def receive_action(data: int = Body(...)):
    print(f"Received action: {data}")
    Action.action=data
    return data


@app.get("/getAction",response_model=Action)
async def getAction():
    return Action