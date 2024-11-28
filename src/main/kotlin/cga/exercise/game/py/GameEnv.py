import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from pydantic import BaseModel

import httpx


timesteps = 5000
apiURL = 'http://127.0.0.1:8000/'


    #FORWARD = 0
    #LEFT= 1
    #BACK = 2
    #RIGHT = 3
    #SHOOT = 4



def sendAction(action):
    action = int(action)
    #print(f'Typ... : {type(pckAction)}')
    try:
        with httpx.Client() as client:
            response = client.post(apiURL + 'sendAction', json=action)
            response.raise_for_status()
            print('sended action...')
    except Exception as e:
        print(f'Error sending action: {action} - {e}')


def fetchGameData():
    try:
        with httpx.Client() as client:
            response = client.get(apiURL + 'get')
            response.raise_for_status()
            data = response.json()



            gameData = {
                'spaceship_position':np.array(data['spaceshipPosition'], dtype=np.float32),
                'spaceship_rotation':np.array(data['spaceshipRotation'], dtype=np.float32),
                'nextAsteroid_position':np.array(data['closestAsteroid'], dtype=np.float32),
            }
            return gameData
        
    except Exception as e:
        print(f'Error: {e}')
        return {
                'spaceship_position': np.zeros(3, dtype=np.float32),
                'spaceship_rotation': np.zeros(1, dtype=np.float32),
                'nextAsteroid_position': np.zeros(3, dtype=np.float32)
            }


class GameEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(GameEnv, self).__init__()

        pos_LOW = np.float32(-1800)
        pos_HIGH = np.float32(1800)
        rot_LOW = np.float32(-np.pi)
        rot_HIGH = np.float32(np.pi)

        # W, A, S, D, P (shift)
        self.action_space = spaces.Discrete(5)

        # spaceship pos, next asteroid pos, spaceship rotation
        self.observation_space = spaces.Dict({
            'spaceship_position': spaces.Box(
                low=np.array([pos_LOW, pos_LOW, pos_LOW]),
                high=np.array([pos_HIGH, pos_HIGH, pos_HIGH]),
                dtype=np.float32
            ),
            'spaceship_rotation': spaces.Box(
                low=np.array([rot_LOW]),
                high=np.array([rot_HIGH]),
                dtype=np.float32
            ),
            'nextAsteroid_position': spaces.Box(
                low=np.array([pos_LOW, pos_LOW,pos_LOW]),
                high=np.array([pos_HIGH, pos_HIGH, pos_HIGH]),
                dtype=np.float32
            )
            # alive
        })

        # inital state
        # get data from fastapi
        self.state = fetchGameData()
        self.done = False 

    def step(self, action):
        reward = 0
        sendAction(action)



        # update state
        self.state = fetchGameData()



        truncated = False
        info = {}



        return self.state, reward, self.done, truncated, info
    
    def reset(self, seed=None, options=None):
        self.done = False
        self.state = fetchGameData()
        info = {}
        return self.state, info
    
    def render(self, mode="human"):
        ...

    def close (self):
        ...


env = GameEnv()
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

check_env(env)  # Check for compliance
model = DQN("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=timesteps)
#tensorboard
new_logger = configure("logs", ["stdout", "tensorboard"])

model.set_logger(new_logger)

