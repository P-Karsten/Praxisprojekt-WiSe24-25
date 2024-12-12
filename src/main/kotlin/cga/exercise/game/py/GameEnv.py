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
from tensorflow.python.eager.context import async_wait

#Run tensor
#tensorboard --logdir=logs/game_rewards/

learningRate = 0.0001
#learningRate = 0.00035
timesteps = 65000
saveInterval = 100000
#eplorationRate = 0.45
max_stepsEpisode = 10000
logname='dqn_spaceship_asteroid_shot-v1'
apiURL = 'http://127.0.0.1:8000/'
log_dir = "logs/game_rewards/" + datetime.datetime.now().strftime("%Y%m%d-%H_%M_%S_ "+logname)
writer = tf.summary.create_file_writer(log_dir)
#RIGHT = 0
#LEFT= 1
#BACK = 2
#FORWARD = 3
#SHOOT = 4

#callback logging / console outputs after 10 episodes
def normalize_angle(angle, range_start=-np.pi, range_end=np.pi):
    range_width = range_end - range_start
    return range_start + ((angle - range_start) % range_width)
def angular_distance(yaw1, yaw2):
    diff = yaw1 - yaw2
    return normalize_angle(diff)
def yaw_distance(yaw1, yaw2):
    """Calculate the shortest distance between two yaw angles."""
    # Normalize to [-pi, pi)
    yaw1 = normalize_angle(yaw1)
    yaw2 = normalize_angle(yaw2)

    # Compute the difference and wrap around
    diff = yaw1 - yaw2
    return normalize_angle(diff)

class gameData: {
    #'spaceship_position': np.zeros(3, dtype=np.float32),
    'spaceship_rotation': np.zeros(1, dtype=np.float32),
    'yaw': np.zeros(1, dtype=np.float32),
    'hit': False,
    'alive': True
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
            #'spaceship_position':np.array(data.get('spaceshipPosition',[0,0,0]), dtype=np.float32),
            'spaceship_rotation':np.array([data.get('spaceshipRotation',0)], dtype=np.float32),
            'yaw':np.array([data.get('yaw',0)],dtype=np.float32),
            'hit': 1 if data.get('hit', False) else 0,
            'alive': 1 if data.get('alive', False) else 0
    
        }
        #print('sended action...',action,"recived:",gameData)
        return gameData
    except Exception as e:
        #print(f'Error sending action: {action} - {e}')
        return {
            #'spaceship_position': np.zeros(3, dtype=np.float32),
            'spaceship_rotation': np.zeros(1, dtype=np.float32),
            'yaw':np.zeros(1, dtype=np.float32),
            'hit': 0,
            'alive': 1,
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
        self.currentaction=6
        self.a0=0
        self.a1=0
        self.a2=0
        self.model = None
        self.hitCounter = 0
        # W, A, S, D, P (shift)
        self.action_space = spaces.Discrete(3)

        # spaceship pos, next asteroid pos, spaceship rotation
        self.observation_space = spaces.Dict({
            #'spaceship_position': spaces.Box(
            #    low=np.array([pos_LOW, pos_LOW, pos_LOW]),
            #    high=np.array([pos_HIGH, pos_HIGH, pos_HIGH]),
            #    dtype=np.float32
            #),
            'spaceship_rotation': spaces.Box(
                low=np.array([rot_LOW]),
                high=np.array([rot_HIGH]),
                dtype=np.float32
            ),
            'yaw': spaces.Box(
                low=np.array([pos_LOW]),
                high=np.array([pos_HIGH]),
                dtype=np.float32
            ),
            #0 false 1 true
            'hit': spaces.Discrete(2),
            'alive': spaces.Discrete(2)
        })

        # inital state
        # get data from fastapi
        self.state = sendAction(10)
        self.done = False

    def setModel(self, model):
        self.model = model

    def step(self, action):

        self.currentaction =action
        gameData = sendAction(self.currentaction)
        rotation = gameData['spaceship_rotation'].item()
        hit = gameData['hit']
        alive = gameData['alive']
        self.reward_ep+=self.reward
        self.reward=0
        # Reward
        #absRotation = abs(rotation)
        #self.reward = -absRotation**2
        #print(math.radians(gameData['yaw'].item()))
        #print(rotation)
        #print((gameData['yaw'].item()))
        yawdistance = yaw_distance(gameData['yaw'].item(),rotation)
        #yawdistance=math.radians(gameData['yaw'].item())-rotation
        #print(yawdistance

        if (alive == 0):
            self.reward -= 10000

        if (hit == 1):
            self.reward+=750
            self.hitCounter+=1
            print('Hit asterioid... reward + 100 !!!!!!')
        if (hit == 0):
            self.reward-=0.25

        if(yawdistance<=1 and yawdistance>=-1):
            if(yawdistance==0.0 or abs(yawdistance)<=0.1):
                self.reward+=13
            else:
                self.reward+=(abs(yawdistance)**-1.1+1)
        else:
            if(abs(yawdistance)>=3):
                self.reward-=4
            else:
                self.reward-=(abs(yawdistance)**2)


        if(self.currentaction==0):
            self.a0+=1
        if(self.currentaction==1):
            self.a1+=1
        if(self.currentaction==2):
            self.a2+=1
        global_step = getattr(self, "step_count", 0)
        self.step_count = global_step + 1
        with writer.as_default():
            tf.summary.scalar("_reward", self.reward_ep, step=global_step)


            if self.model and math.fmod(self.step_count,500)==0:
                tf.summary.scalar("exploration", self.model.exploration_rate , step=global_step)
                tf.summary.scalar("learning_rate", self.model.learning_rate , step=global_step)
                tf.summary.scalar("gamma", self.model.gamma , step=global_step)



        if  (alive == 0 or self.hitCounter >= 5):
            with writer.as_default():
                tf.summary.scalar("reward_ep", self.reward_ep/max_stepsEpisode, step=global_step)
                tf.summary.scalar("action_0", self.a0, step=global_step)
                tf.summary.scalar("action_1", self.a1, step=global_step)
                tf.summary.scalar("action_2", self.a2, step=global_step)
                self.a0=0
                self.a1=0
                self.a2=0
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
        self.hitCounter = 0
        self.done = False
        self.reward = 0
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


#Training functions
def modelTrain(env: GameEnv, modelName: str, exp: float, totalSteps: int):
    model = DQN.load(modelName, env=env)
    env.setModel(model)
    model.exploration_initial_eps = exp
    model.learn(total_timesteps=totalSteps, log_interval=5)
    model.save(modelName)

def modelInit(env: GameEnv, modelName: str, expInit: float, expFinal: float, expFrac: float,  totalSteps: int, lr: float):
    model = DQN("MultiInputPolicy", env, verbose=2, exploration_initial_eps=expInit, exploration_final_eps=expFinal, exploration_fraction=expFrac, learning_rate=lr)
    env.setModel(model)
    model.buffer_size = 50000
    model.batch_size=64
    model.gamma = 0.99
    model.learn(total_timesteps=totalSteps, log_interval=5)
    model.save(modelName)

def modelTrainAutomatic(env: GameEnv, modelName: str, expInit: float, expFinal: float, expFrac: float, totalSteps: int, cycles: int):
    x = 0
    while x < cycles:
        model = DQN.load(modelName, env=env)
        env.setModel(model)
        model.exploration_initial_eps = expInit
        model.exploration_final_eps = expFinal
        model.exploration_fraction = expFrac
        model.buffer_size = 50000
        model.learn(total_timesteps=totalSteps, log_interval=1)
        model.save(modelName)
        print('Model saved...')

        x += 1
        print("cycle start...",x)

def modelPredict(env: GameEnv, modelName: str, episodes: int):
    model = DQN.load(modelName, env=env)
    state, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(state)
        state, reward, done, truncated, info = env.step(action)




#Training:
#modelInit(env,logname,0.8,0.1,0.5,500000,0.001)
modelInit(env,logname,0.8,0.1,0.5,500000,0.00025)

#modelTrainAutomatic(env, logname, 0.3,0.1,0.5, 50000, 5)