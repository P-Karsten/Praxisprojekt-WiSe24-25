# Install FastAPI and Uvicorn if you haven't
# pip install fastapi uvicorn
#Ausführen in py ordner     python -m uvicorn test:app --reload
from fastapi import FastAPI
import random
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import DQN
app = FastAPI()

goal=10000
# Define the GameData model
class Vector3f1(BaseModel):
    x: float
    y: float
    z: float

class GameData(BaseModel):
    spaceshipPosition: List[float]
    spaceshipRotation: Vector3f1
    closestAsteroid: Vector3f1
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
        closestAsteroid=data.closestAsteroid,
        reward=data.reward,
        time=data.time
    )
    return response_data

"""class CustomEnv(gym.Env):
    #Custom Environment that follows gym interface.

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        spaceshipPosition = np.array([0, 0, 0])
        # Get spaceship rotation
        spaceshipRotation = np.array([0])
        # Get asteroid coordinates
        nextAsterioidPosition = np.array([0, 0, 0])
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(5)
        # Example for using image as input (channel-first; channel-last also works):
        low=np.full(7, -1700, dtype=np.float32)
        high=np.full(7, 1700, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, action):
        ...
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        ...
        return observation, info

    def render(self):
        ...

    def close(self):
        ...
    # Instantiate the env
env = CustomEnv(arg1, ...)
# Define and Train the agent
model = DQN("MlpPolicy", env).learn(total_timesteps=1000)"""
