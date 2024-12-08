import asyncio
import datetime
import math
import time
from typing import List
import tensorflow as tf
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from pydantic import BaseModel
import httpx


#Run tensor
#tensorboard --logdir=logs/game_rewards/

learningRate = 0.0001
#learningRate = 0.00035
timesteps = 65000
saveInterval = 100000
#eplorationRate = 0.45
max_stepsEpisode = 5000

apiURL = 'http://127.0.0.1:8000/'
log_dir = "logs/game_rewards/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(log_dir)

#RIGHT = 0
#LEFT= 1
#BACK = 2
#FORWARD = 3
#SHOOT = 4

#callback logging / console outputs after 10 episodes


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
        #print('sended action...',action,"recived:",gameData)
        return gameData
    except Exception as e:
        #print(f'Error sending action: {action} - {e}')
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
        self.reward=0.0
        self.reward_ep=0.0
        pos_LOW = np.float32(-1800)
        pos_HIGH = np.float32(1800)
        rot_LOW = np.float32(-np.pi)
        rot_HIGH = np.float32(np.pi)

        self.model = None
        # W, A, S, D, P (shift)
        self.action_space = spaces.Discrete(3)

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

    def setModel(self, model):
        self.model = model

    def step(self, action):

        gameData = sendAction(action)
        rotation = gameData['spaceship_rotation'].item()
        self.reward_ep+=self.reward
        self.reward=0
        # Reward
        #absRotation = abs(rotation)
        #self.reward = -absRotation**2


        if(rotation<=1 and rotation>=-1):
            if(rotation==0.0 or abs(rotation)<=0.1):
                self.reward+=50
            else:
                self.reward+=(abs(rotation)**-1.1+2)
        else:
            if(abs(rotation)>=3):
                self.reward-=15
            else:
                self.reward-=(abs(rotation)**2)



        global_step = getattr(self, "step_count", 0)
        with writer.as_default():
            tf.summary.scalar("reward", self.reward_ep, step=global_step)


            if self.model:
                tf.summary.scalar("exploration", self.model.exploration_rate , step=global_step)

        self.step_count = global_step + 1


        if  math.fmod(self.step_count, max_stepsEpisode) == 0:
            with writer.as_default():
                tf.summary.scalar("reward_ep", self.reward_ep/max_stepsEpisode, step=global_step)
            self.reward_ep=0
            self.done = True

        self.state = gameData
        #self.done = False
        truncated = False
        info = {}

        return self.state, self.reward, self.done, truncated, info


    def reset(self, seed=None, options=None):
        if self.done:
            sendAction(10)
            asyncio.sleep(2)
            print('Game reset...')
        self.done = False
        self.reward = 0
        print('Episode finished...')
        self.state = sendAction(6)
        info = {}
        return self.state, info

    def render(self, mode="human"):
        ...

    def close (self):
        ...



#Create/check env instance
env = GameEnv()
check_env(env)

#Working exploration values
"""
model.exploration_initial_eps = 0.375
model.exploration_final_eps = 0.275
model.exploration_fraction = 0.5
"""

#constant train run ideal (0.1 - 0.05 later)
#model.exploration_initial_eps = 0.15
#model.exploration_final_eps = 0.15
#model.exploration_fraction = 0.6


"""
#final
model.exploration_initial_eps = 0.1
model.exploration_final_eps = 0.05
model.exploration_fraction = 0.3
"""


#Training functions
def modelTrain(env: GameEnv, modelName: str, exp: float, totalSteps: int):
    model = DQN.load(modelName, env=env)
    env.setModel(model)
    model.exploration_initial_eps = exp
    model.learn(total_timesteps=totalSteps, log_interval=5)
    model.save(modelName)

def modelInit(env: GameEnv, modelName: str, expInit: float, expFinal: float, expFrac: float,  totalSteps: int, lr: float):
    model = DQN("MultiInputPolicy", env, verbose=2, exploration_initial_eps=expInit, exploration_final_eps=expFinal, exploration_fraction=expFrac, learning_rate=lr, tensorboard_log="./logs/game_rewards/")
    env.setModel(model)
    model.learn(total_timesteps=totalSteps, log_interval=5)
    model.save(modelName)

def modelTrainAutomatic(env: GameEnv, modelName: str, expInit: float, expFinal: float, expFrac: float, totalSteps: int, cycles: int):
    x = 0
    while x <= cycles:
        model = DQN.load(modelName, env=env)
        env.setModel(model)
        model.exploration_initial_eps = expInit
        model.exploration_final_eps = expFinal
        model.exploration_fraction = expFrac
        model.buffer_size = 50000
        model.learn(total_timesteps=totalSteps, log_interval=5)
        model.save(modelName)
        print('Model saved...')

        x += 1
        print("cycle start...",x)




#Training:
modelInit(env,"dqn_spaceship_3actionsv2",0.7,0.2,0.6,200000,0.0003)
#modelTrainAutomatic(env, 'dqn_spaceship_3actionsv2', 0.6, 50000, 5)