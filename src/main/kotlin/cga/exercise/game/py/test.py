# pip install fastapi uvicorn
#Ausführen in py ordner     python -m uvicorn test:app --reload
import asyncio
import time
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

from fastapi import Body


app = FastAPI()

class Vector3f1(BaseModel):
    x: float
    y: float
    z: float

class GameData(BaseModel):
    spaceshipPosition: List[float]
    spaceshipRotation: Vector3f1
    yaw : float
    hit: bool
    alive: bool

global starttime
starttime=time.time()
global starttime2
starttime2=time.time()
ready_to_send = asyncio.Event()
ready_to_action = asyncio.Event()
class Action(BaseModel):
    action: int =6

savedData: GameData =None


@app.on_event("startup")
async def startup_event():
    ready_to_action.set()


latest_action = 6

@app.post("/send/", response_model=Action)
async def receive_game_data(data: GameData):
    global savedData, starttime, latest_action

    await ready_to_action.wait()

    response_data = GameData(
        spaceshipPosition=[pos for pos in data.spaceshipPosition],
        spaceshipRotation=data.spaceshipRotation,
        yaw=data.yaw,
        hit=data.hit,
        alive=data.alive
    )
    savedData = response_data
    print(savedData.yaw)
    print(f"time send : {time.time() - starttime:.3f}sec")
    starttime = time.time()

    ready_to_send.set()
    ready_to_action.clear()

    # Return the latest action
    return Action(action=latest_action)



@app.post("/sendAction")
async def receive_action(data: int = Body(...)):
    global starttime2, latest_action

    await ready_to_send.wait()

    latest_action = data  # Update the global variable
    print(f"time action : {time.time() - starttime2:.3f}sec")
    starttime2 = time.time()

    ready_to_action.set()
    ready_to_send.clear()

    if savedData is None:
        return {'Error': 'No data received'}

    return {
        "spaceshipPosition": savedData.spaceshipPosition,
        "spaceshipRotation": savedData.spaceshipRotation.y,
        "yaw": savedData.yaw,
        "hit": savedData.hit,
        "alive": savedData.alive
    }