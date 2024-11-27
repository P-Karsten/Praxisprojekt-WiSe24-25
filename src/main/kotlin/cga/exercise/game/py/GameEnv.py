import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN

import httpx

GOAL = 10000
timesteps = 500
apiURL = 'http://127.0.0.1:8000/'


    #LEFT = 0
    #RIGHT = 1
    #FORWARD = 2
    #BACK = 3
    #SHOOT = 4



def sendAction(action):
    actionInt = int(action)
    print(f'Typ... : {type(actionInt)}')
    try:
        with httpx.Client() as client:
            response = client.post(apiURL + 'sendAction', json=action)
            response.raise_for_status()
            print('sended action...')
    except Exception as e:
        print(f'Error sending action: {actionInt} - {e}')


def fetchGameData():
    try:
        with httpx.Client() as client:
            response = client.get(apiURL + 'get')
            response.raise_for_status()
            data = response.json()

            spaceship_rotation = np.array([data['spaceshipRotation']], dtype=np.float32) 

            gameData = {
                'spaceship_position':np.array(data['spaceshipPosition'], dtype=np.float32),
                'spaceship_rotation': spaceship_rotation,
                #'spaceship_rotation':np.array(data['spaceshipRotation'], dtype=np.float32),
                'nextAsterioid_position':np.array(data['closestAsterioid'], dtype=np.float32),
            }
            return gameData
        
    except Exception as e:
        print(f'Error: {e}')
        return {
                'spaceship_position': np.zeros(3, dtype=np.float32),
                'spaceship_rotation': np.zeros(1, dtype=np.float32),
                'nextAsterioid_position': np.zeros(3, dtype=np.float32)
            }


class GameEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {"render.modes": ["human"], 'render_fps': 25}

    def __init__(self):
        super(GameEnv, self).__init__()

        pos_LOW = np.float32(-1800)
        pos_HIGH = np.float32(1800)
        rot_LOW = np.float32(-np.pi)
        rot_HIGH = np.float32(np.pi)

        # W, A, S, D, P (shift)
        self.action_space = spaces.Discrete(5)

        # spaceship pos, next asteriood pos, spaceship rotation
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
            'nextAsterioid_position': spaces.Box(
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
        print(f'Action: {action}')

        # update state
        self.state = fetchGameData()


        # Logic to compute the reward and done state can go here
        if np.linalg.norm(self.state['spaceship_position']) < 10:
            reward = 10  # Example: Reward for getting close to the asteroid
        else:
            reward = -1  # Penalty for other actions
        self.done = reward >= GOAL  # Episode ends when GOAL is achieved

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
check_env(env)

model = DQN('MbpPolicy', env, verbose=1)
model.learn(total_timesteps=timesteps)

