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
import optuna
#Run tensor
#tensorboard --logdir=logs/game_rewards/
from stable_baselines3 import A2C
maxScore = 15
learningRate = 0.0001
#learningRate = 0.00035
timesteps = 65000
saveInterval = 100000
#eplorationRate = 0.45
max_stepsEpisode = 10000
logname='A2C_spaceship_asteroid_shot_l_yaw_pitch_fix_v1'
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
    #'spaceship_rotation': np.zeros(3, dtype=np.float32),
    'pitch': np.zeros(1, dtype=np.float32),
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
            #'spaceship_rotation':np.array(data.get('spaceshipPosition',[0,0,0]), dtype=np.float32),
            'pitch':np.array([data.get('pitch',0)],dtype=np.float32),
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
            #'spaceship_rotation': np.zeros(3, dtype=np.float32),
            'pitch':np.zeros(1, dtype=np.float32),
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
        self.high_score=0.0
        self.short_ep=max_stepsEpisode*500
        pos_LOW = np.float32(-1800)
        pos_HIGH = np.float32(1800)
        rot_LOW = np.float32(-np.pi)
        rot_HIGH = np.float32(np.pi)
        self.currentaction=10
        self.a0=0
        self.a1=0
        self.a2=0
        self.a3=0
        self.a4=0
        self.previous_yawdistance =3.14
        self.previous_pitchdistance=3.14
        self.model = None
        self.hitCounter = 0
        # W, A, S, D, P (shift)
        self.action_space = spaces.Discrete(5)

        # spaceship pos, next asteroid pos, spaceship rotation
        self.observation_space = spaces.Dict({
            #'spaceship_position': spaces.Box(
            #    low=np.array([pos_LOW, pos_LOW, pos_LOW]),
            #    high=np.array([pos_HIGH, pos_HIGH, pos_HIGH]),
            #    dtype=np.float32
            #),
            #'spaceship_rotation': spaces.Box(
            #   low=np.array([rot_LOW, rot_LOW, rot_LOW]),
            #  high=np.array([rot_HIGH, rot_HIGH, rot_HIGH]),
            # dtype=np.float32
            #),
            'pitch': spaces.Box(
                low=np.array([rot_LOW]),
                high=np.array([rot_HIGH]),
                dtype=np.float32
            ),
            'yaw': spaces.Box(
                low=np.array([rot_LOW]),
                high=np.array([rot_HIGH]),
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

        gameData = sendAction(self.currentaction)
        self.currentaction =action
        hit = gameData['hit']
        alive = gameData['alive']
        self.reward_ep+=self.reward
        self.reward=0
        self.state = {
            #'spaceship_rotation':gameData['spaceship_rotation'],
            'pitch': gameData['pitch'],
            'yaw': gameData['yaw'],
            'hit': gameData['hit'],
            'alive': gameData['alive']
        }
        yawdistance = gameData['yaw'].item()
        pitchdistance = gameData['pitch'].item()
        #print(pitchdistance)
        self.reward-=0.25
        if (alive == 0):
            self.reward -= 5000
            #print('dead...')

        if (self.hitCounter >= maxScore):
            self.reward += 50000000/((self.ep_step)**0.85)
        if(hit==1):
            self.hitCounter+=1
            self.reward=10
        if (hit == 1 and abs(yawdistance)<=0.05 and abs(pitchdistance)<=0.05):
            self.reward+=1000
            #self.reward+=1500*self.hitCounter
            #print('Hit asterioid...',self.hitCounter)
        if(self.currentaction==2):
            self.reward-=0.5


        #if(abs(yawdistance)<=1):
        if((yawdistance==0.0 or abs(yawdistance)<=0.035) and (pitchdistance==0.0 or abs(pitchdistance)<=0.035)):
            self.reward+=1
            #if(self.currentaction==2):
            #self.reward+=100
            #self.reward+=500 #PPO

        self.reward-=(abs(yawdistance))
        self.reward-=(abs(pitchdistance))
        if(abs(yawdistance)<=0.035 and abs(pitchdistance)<=0.035 and self.currentaction==2):
            self.reward+=3
        if abs(yawdistance) < abs(self.previous_yawdistance):
           self.reward += 0.11
        if abs(pitchdistance) < abs(self.previous_pitchdistance):
           self.reward += 0.1
        # if(pitchdistance==0.0 or abs(pitchdistance)<=0.025):
        #    self.reward+=10
        #if(self.currentaction==2):
        #    self.reward+=3
        #else:
        #   self.reward-=(abs(pitchdistance)*2)
        #self.reward+=(abs(yawdistance)**-0.4)
        """else:
            if(abs(yawdistance)>=3):
                self.reward-=2
            else:
                self.reward-=2*(abs(yawdistance)**0.5-1)"""


        if(self.currentaction==0):
            self.a0+=1
        if(self.currentaction==1):
            self.a1+=1
        if(self.currentaction==2):
            self.a2+=1
        if(self.currentaction==3):
            self.a3+=1
        if(self.currentaction==4):
            self.a4+=1
        global_step = getattr(self, "step_count", 0)
        self.step_count = global_step + 1
        with writer.as_default():
            tf.summary.scalar("_reward", self.reward_ep, step=global_step)


            if self.model and math.fmod(self.step_count,500)==0:
                #tf.summary.scalar("exploration", self.model.exploration_rate , step=global_step)
                tf.summary.scalar("learning_rate", self.model.learning_rate , step=global_step)
                tf.summary.scalar("gamma", self.model.gamma , step=global_step)



        if  (alive == 0 or self.hitCounter >= maxScore):
            with writer.as_default():
                tf.summary.scalar("reward_ep", self.reward_ep/self.ep_step, step=global_step)
                tf.summary.scalar("score", self.hitCounter, step=global_step)
                tf.summary.scalar("action_0%", self.a0/self.ep_step, step=global_step)
                tf.summary.scalar("action_1%", self.a1/self.ep_step, step=global_step)
                tf.summary.scalar("action_2%", self.a2/self.ep_step, step=global_step)
                tf.summary.scalar("action_3%", self.a3/self.ep_step, step=global_step)
                tf.summary.scalar("action_4%", self.a4/self.ep_step, step=global_step)
                tf.summary.scalar("ep_length", self.ep_step, step=global_step)
                print(self.hitCounter)
                self.a0=0
                self.a1=0
                self.a2=0
                self.a3=0
                self.a4=0
                self.model.save(logname+"_A")
            if(self.ep_step<=self.short_ep and self.hitCounter>=maxScore):
                self.model.save(logname+"_short")
                self.short_ep=self.ep_step
                print("saved...",self.ep_step,"global step:",global_step)
            self.reward_ep=0
            self.done = True

        #self.state = gameData
        #print(f"State: {self.state}, Predicted Action: {action}",self.reward)
        self.ep_step+=1
        self.previous_yawdistance=yawdistance
        self.previous_pitchdistance=pitchdistance

        #self.done = False
        truncated = False
        info = {}

        return self.state, self.reward, self.done, truncated, info


    def reset(self, seed=None, options=None):
        self.hitCounter = 0
        self.reward = 0
        gameData = sendAction(10)
        self.state={
            #'spaceship_rotation':gameData['spaceship_rotation'],
            'pitch': gameData['pitch'],
            'yaw': gameData['yaw'],
            'hit': gameData['hit'],
            'alive': gameData['alive']
        }
        self.reward = 0
        self.ep_step=0
        self.done = False
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

def modelInit(env: GameEnv, modelName: str,  totalSteps: int, lr: float):
    model = A2C("MultiInputPolicy", env, verbose=2, learning_rate=lr, device="cpu")
    #model.use_sde=True
    env.setModel(model)
    model.n_steps=5
    #model.buffer_size = 1000000
    #model.batch_size=64
    model.gamma = 0.95
    model.gae_lambda = 1.0
    model.ent_coef = 0.0
    model.vf_coef = 0.5
    model.max_grad_norm = 0.5
    model.rms_prop_eps = 1e-5
    model.use_rms_prop = True
    model.normalize_advantage = False
    model.stats_window_size = 100
    model.verbose = 0


#model.tau=0.10
    model.learn(total_timesteps=totalSteps, log_interval=1)
    model.save(modelName)

def modelTrainAutomatic(env: GameEnv, modelName: str, expInit: float, expFinal: float, expFrac: float, totalSteps: int, cycles: int):
    x = 0
    while x < cycles:
        model = DQN.load(modelName, env=env, device='cuda')
        env.setModel(model)
        model.exploration_initial_eps = expInit
        model.exploration_final_eps = expFinal
        model.exploration_fraction = expFrac
        model.buffer_size = 1000000
        model.learn(total_timesteps=totalSteps, log_interval=1)
        print('Model saved...')
        model.save(modelName)
        x += 1
        print("cycle start...",x)

def modelPredict(env: GameEnv, modelName: str, episodes: int):
    # Load the trained model
    model = DQN.load(modelName, env=env)
    env.setModel(model)

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Ensure state is in the correct format for the model
            action, _ = model.predict(state, deterministic=False)  # Deterministic mode for evaluation
            state, reward, done, truncated, info = env.step(action)
            print(f"State: {state}, Predicted Action: {action}",reward)
            # Accumulate rewards
            total_reward += reward

        print(f"Episode {episode + 1}/{episodes}: Total Reward = {total_reward}")


#Training:
#modelInit(env,logname,0.8,0.1,0.5,500000,0.001)
modelInit(env,logname,1000000,0.00025)#todo rotations beschleunigung zb. 20 gleiche inputs schneller drehen #todo only prev_dis reward ??
#modelPredict(env,logname,1)
#modelTrainAutomatic(env, logname, 0.3,0.05,0.7, 1000000, 1)

