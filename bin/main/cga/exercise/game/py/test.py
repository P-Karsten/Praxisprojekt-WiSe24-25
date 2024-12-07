# Install FastAPI and Uvicorn if you haven't
# pip install fastapi uvicorn
#Ausführen in py ordner     python -m uvicorn test:app --reload
import time
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

global starttime
starttime=time.time()
global starttime2
starttime2=time.time()

class Action(BaseModel):
    action: int =6

savedData: GameData =None

# Endpoint to handle POST requests with GameDatas
@app.post("/send/",response_model=Action)
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
    global starttime
    print(f"time send : {time.time()-starttime:.3f}sec")
    starttime=time.time()
    return Action


@app.post("/sendAction")
async def receive_action(data: int = Body(...)):
    #print(f"Received action: {data}")
    Action.action=data
    global starttime2
    print(f"time action : {time.time()-starttime2:.3f}sec")
    starttime2=time.time()
    time.sleep(0.010)
    if savedData is None:
        return {'Error: No data received'}
    return {
        "spaceshipPosition": savedData.spaceshipPosition,
        "spaceshipRotation": savedData.spaceshipRotation.y,
        "closestAsteroid": [
            savedData.closestAsteroid.x,
            savedData.closestAsteroid.y,
            savedData.closestAsteroid.z
        ]
    }
