import asyncio
import datetime
from typing import List
import tensorflow as tf
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
log_dir = "logs/game_rewards/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(log_dir)

    #FORWARD = 0
    #LEFT= 1
    #BACK = 2
    #RIGHT = 3
    #SHOOT = 4

class gameData: {
    'spaceship_position': np.zeros(3, dtype=np.float32),
    'spaceship_rotation': np.zeros(1, dtype=np.float32),
    'nextAsteroid_position': np.zeros(3, dtype=np.float32)
}
client=httpx.Client(http2=True)
def sendAction(action):
    action = int(action)
    #print(f'Typ... : {type(pckAction)}')
    try:

            response = client.post(f"{apiURL}sendAction", json=action)
            response.raise_for_status()
            data=response.json()
            gameData = {
                'spaceship_position':np.array(data.get('spaceshipPosition',[0,0,0]), dtype=np.float32),
                'spaceship_rotation':np.array([data.get('spaceshipRotation',0)], dtype=np.float32),
                'nextAsteroid_position':np.array(data.get('closestAsteroid',[0,0,0]), dtype=np.float32),
            }
            print('sended action...',action,"recived:",gameData)
            return gameData
    except Exception as e:
        print(f'Error sending action: {action} - {e}')
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
        self.state = sendAction(6)
        self.done = False 

    def step(self, action):

        gameData = sendAction(action)
        print('spaceshiprotation:' ,gameData['spaceship_rotation'])
        if(gameData['spaceship_rotation']<=0.5 and gameData['spaceship_rotation']>=-0.5):
            reward=5
            print("reward+",reward)
        else:
            reward=-5
            print("reward",reward)

        global_step = getattr(self, "step_count", 0)
        with writer.as_default():
            tf.summary.scalar("reward", reward, step=global_step)

        # Schrittzähler aktualisieren
        self.step_count = global_step + 1
        # update state
        self.state = gameData
        #print(self.state)


        truncated = False
        info = {}



        return self.state, reward, self.done, truncated, info
    
    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.done = False
        self.state = sendAction(6)
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

