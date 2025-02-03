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
import matplotlib.pyplot as plt
import io
from tensorflow.python.eager.context import async_wait
import graphics as Graphics
import psutil
import ram as Ram
#Run tensor
#tensorboard --logdir=logs/game_rewards/
predictv=True
maxScore = 30
learningRate = 0.0001
#learningRate = 0.00035
timesteps = 65000
saveInterval = 100000
#eplorationRate = 0.45
max_stepsEpisode = 10000
logname='dqn_spaceship_asteroid_shot_FinalGame-v22'
apiURL = 'http://127.0.0.1:8000/'
log_dir = "logs/game_rewards/"+logname+"/" + datetime.datetime.now().strftime("%d.%m.%Y--%H_%M_%S")
writer = tf.summary.create_file_writer(log_dir)
#RIGHT = 0
#LEFT= 1
#BACK = 2
#FORWARD = 3
#SHOOT = 4

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
    'posDistance': np.zeros(1, dtype=np.float32),
    'pitch': np.zeros(1, dtype=np.float32),
    'yaw': np.zeros(1, dtype=np.float32),
    'hit': np.zeros(1, dtype=np.float32),
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
            'posDistance': np.array([data.get('posDistance',0)],dtype=np.float32),
            'pitch':np.array([data.get('pitch',0)],dtype=np.float32),
            'yaw':np.array([data.get('yaw',0)],dtype=np.float32),
            'hit': np.array([data.get('counter',0)],dtype=np.float32),
            'alive': 1 if data.get('alive', False) else 0
        }
        #print('sended action...',action,"recived:",gameData)
        return gameData
    except Exception as e:
        #print(f'Error sending action: {action} - {e}')
        return {
            #'spaceship_position': np.zeros(3, dtype=np.float32),
            #'spaceship_rotation': np.zeros(3, dtype=np.float32),
            'posDistance': np.zeros(1, dtype=np.float32),
            'pitch':np.zeros(1, dtype=np.float32),
            'yaw':np.zeros(1, dtype=np.float32),
            'hit': np.zeros(1, dtype=np.float32),
            'alive': 1,
        }


class GameEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(GameEnv, self).__init__()
        self.predictv=False
        self.reward=0.0
        self.reward_ep=0.0
        self.high_score=0.0
        self.short_ep=max_stepsEpisode*500
        pos_LOW = np.float32(0)
        pos_HIGH = np.float32(1800)
        rot_LOW = np.float32(-np.pi)
        rot_HIGH = np.float32(np.pi)
        self.currentaction=10
        self.hit=0
        self.prevhit=0
        self.a0=0
        self.a1=0
        self.a2=0
        self.a3=0
        self.a4=0
        self.a5=0
        self.a6=0

        #Reward stats
        self.reward_shot=0
        self.reward_aim=0
        self.reward_w=0
        self.reward_s=0
        self.reward_death=0
        self.reward_hits=0
        self.reward_goal=0
        self.reward_exist=0

        self.previous_yawdistance =3.14
        self.previous_pitchdistance=3.14
        self.model = None
        self.hitCounter = 0
        # W, A, S, D, P (shift)
        self.action_space = spaces.Discrete(7)

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
            'posDistance': spaces.Box(
                low=np.array([pos_LOW]),
                high=np.array([pos_HIGH]),
                dtype=np.float32
            ),
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
            'hit': spaces.Box(
                low=np.array([0]),
                high=np.array([maxScore]),
                dtype=np.float32
            ),
            #0 false 1 true
            #'hit': spaces.Discrete(2),
            'alive': spaces.Discrete(2),
        })

        # inital state
        # get data from fastapi
        self.state = sendAction(10)
        self.done = False

    def setModel(self, model):
        self.model = model

    def step(self, action):
        gameData = sendAction(self.currentaction)
        self.currentaction=action
        hit = gameData['hit']
        alive = gameData['alive']
        self.reward_ep+=self.reward
        self.reward=0
        self.state = {
            'posDistance': gameData['posDistance'],
            'pitch': gameData['pitch'],
            'yaw': gameData['yaw'],
            'hit': gameData['hit'],
            'alive': gameData['alive']
        }
        yawdistance = gameData['yaw'].item()
        pitchdistance = gameData['pitch'].item()
        astDistance = gameData['posDistance']

        #print(astDistance)
        self.reward-=0.5
        self.reward_exist-=0.5
        if (alive == 0):
            self.reward -= 2000
            self.reward_death -= 2000
        if (self.hit >= maxScore):
            self.reward += 50000000/((self.ep_step)**0.85)
            self.reward_goal += 50000000/((self.ep_step)**0.85)

        if(hit > self.prevhit and abs(yawdistance)<=0.045 and abs(pitchdistance)<=0.045):
            self.reward+=500
            self.reward_hits+=500

        if(hit>self.prevhit):
            self.hitCounter+=1
            self.reward+=100
            self.reward_hits+=100
            self.prevhit=hit

        if((yawdistance==0.0 or abs(yawdistance)<=0.03) and (pitchdistance==0.0 or abs(pitchdistance)<=0.03)):
            self.reward+=1.5
            self.reward_aim+=1.5

        if((yawdistance==0.0 or abs(yawdistance)<=0.075) and (pitchdistance==0.0 or abs(pitchdistance)<=0.075)):
            if(astDistance <= 200 and self.currentaction==5):
                self.reward+=0.5
                self.reward_s+=0.5
            if(astDistance >= 600 and self.currentaction==5):
                self.reward-=0.6
                self.reward_s-=0.6
            if(astDistance >= 800 and self.currentaction==6):
                self.reward+=1
                self.reward_w+=1
            if(astDistance <= 250 and self.currentaction==6):
                self.reward-=1.5
                self.reward_w-=1.5

        self.reward-=(abs(yawdistance))*0.2
        self.reward_aim-=(abs(yawdistance))*0.2
        self.reward-=(abs(pitchdistance))*0.2
        self.reward_aim-=(abs(pitchdistance))*0.2
        #if((yawdistance==0.0 or abs(yawdistance)<=0.025) and (pitchdistance==0.0 or abs(pitchdistance)<=0.025) and astDistance<=800 and self.currentaction==2):
            #self.reward+=500
            #self.reward_shot+=500
        if((yawdistance==0.0 or abs(yawdistance)<=0.033) and (pitchdistance==0.0 or abs(pitchdistance)<=0.033) and astDistance<=800 and self.currentaction==2):
            self.reward+=5
            self.reward_shot+=5
        #if((yawdistance==0.0 or abs(yawdistance)<=0.025) and (pitchdistance==0.0 or abs(pitchdistance)<=0.025) and astDistance>=800 and self.currentaction==2):
            #self.reward-=15
            #self.reward_shot-=15
        if abs(yawdistance) < abs(self.previous_yawdistance) and (self.currentaction==0 or self.currentaction==1) and abs(yawdistance)>0.01:
            self.reward += 0.1
            self.reward_aim += 0.1
            #print("yaw before",(abs(self.previous_yawdistance),abs(yawdistance)))
        if abs(pitchdistance) < abs(self.previous_pitchdistance) and (self.currentaction==3 or self.currentaction==4)and abs(pitchdistance)>0.01:
            self.reward += 0.1
            self.reward_aim += 0.1
            #print("pitch before",(abs(self.previous_pitchdistance),abs(pitchdistance)))
        if abs(yawdistance) > abs(self.previous_yawdistance) and (self.currentaction==0 or self.currentaction==1) and abs(yawdistance)>0.01:
            self.reward -= 0.1
            self.reward_aim -= 0.1
            #print("yaw before",(abs(self.previous_yawdistance),abs(yawdistance)))
        if abs(pitchdistance) > abs(self.previous_pitchdistance) and (self.currentaction==3 or self.currentaction==4)and abs(pitchdistance)>0.01:
            self.reward -= 0.1
            self.reward_aim -= 0.1

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
        if(self.currentaction==5):
            self.a5+=1
        if(self.currentaction==6):
            self.a6+=1
        global_step = getattr(self, "step_count", 0)
        self.step_count = global_step + 1
        with writer.as_default():
            tf.summary.scalar("Reward", self.reward_ep, step=global_step)
            if self.model and math.fmod(self.step_count,500)==0:
                with tf.name_scope("Hyperparameter"):
                    tf.summary.scalar("Exploration", self.model.exploration_rate , step=global_step)
                    tf.summary.scalar("Learning Rate", self.model.learning_rate , step=global_step)
                    tf.summary.scalar("Gamma", self.model.gamma , step=global_step)
                    tf.summary.scalar("Tau", self.model.tau , step=global_step)

        if  (alive == 0 or self.hitCounter >= maxScore):
            with writer.as_default():
                with tf.name_scope("General"):
                    tf.summary.scalar("Reward_EP", self.reward_ep / self.ep_step, step=global_step)
                    tf.summary.scalar("Score", self.hitCounter, step=global_step)
                    tf.summary.scalar("Episode length", self.ep_step, step=global_step)
                    process_ram_usage = Ram.get_process_ram_usage("java.exe")
                    tf.summary.scalar(f"{"java.exe"} RAM Usage (MB)", process_ram_usage, step=global_step)

                total_reward = self.reward_shot + self.reward_aim + self.reward_death + self.reward_hits + self.reward_goal + self.reward_w + self.reward_s + self.reward_exist
                total_actions = self.a0 + self.a1 + self.a2 +self.a3 + self.a4 + self.a5 + self.a6

                rewards_dict = {
                    "Shot": self.reward_shot,
                    "Aim": self.reward_aim,
                    "Death": self.reward_death,
                    "Asteroid hits": self.reward_hits,
                    "Reaching goal": self.reward_goal,
                    "W": self.reward_w,
                    "S": self.reward_s,
                    "Agent exists": self.reward_exist,
                }
                actions_dict = {
                    "Action_0": self.a0,
                    "Action_1": self.a1,
                    "Action_2": self.a2,
                    "Action_3": self.a3,
                    "Action_4": self.a4,
                    "Action_5": self.a5,
                    "Action_6": self.a6,
                }
                percentage_actions = {k: (v / total_actions) * 100 if total_actions != 0 else 0 for k, v in actions_dict.items()}

                #actions pie chart
                action_chart = Graphics.plot_action_chart(actions_dict, percentage_actions, total_actions, global_step)
                tf.summary.image("Actions per episode", action_chart, step=global_step)
                #reward diagram
                reward_diagram = Graphics.plot_reward_diagram(rewards_dict, total_reward, global_step)
                tf.summary.image("Rewards per Episode", reward_diagram, step=global_step)

                print(self.hitCounter)
                self.a0=0
                self.a1=0
                self.a2=0
                self.a3=0
                self.a4=0
                self.a5=0
                self.a6=0
                self.reward_shot=0
                self.reward_aim=0
                self.reward_w=0
                self.reward_s=0
                self.reward_death=0
                self.reward_hits=0
                self.reward_goal=0
                self.reward_exist=0
                self.model.save("DQN/"+logname)
            if(self.ep_step<=self.short_ep and self.hitCounter>=maxScore):
                self.model.save("DQN/"+logname+"_short")
                self.short_ep=self.ep_step
                #print("saved...",self.ep_step,"global step:",global_step)
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
        self.prevhit=0
        self.reward = 0
        gameData = sendAction(10)
        self.state={
            #'spaceship_rotation':gameData['spaceship_rotation'],
            'posDistance': gameData['posDistance'],
            'pitch': gameData['pitch'],
            'yaw': gameData['yaw'],
            'hit': gameData['hit'],
            'alive': gameData['alive'],
        }
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
def modelInit(env: GameEnv, modelName: str, expInit: float, expFinal: float, expFrac: float,  totalSteps: int, lr: float):
    env.predictv = False
    model = DQN("MultiInputPolicy", env, verbose=2, exploration_initial_eps=expInit, exploration_final_eps=expFinal, exploration_fraction=expFrac, learning_rate=lr, device="cuda")
    env.setModel(model)
    model.buffer_size = 1000000
    model.batch_size=128
    model.gamma = 0.98
    model.tau=0.0025
    model.learn(total_timesteps=totalSteps, log_interval=1)
    model.save("DQN/"+modelName)

def modelTrainAutomatic(env: GameEnv, modelName: str, expInit: float, expFinal: float, expFrac: float, totalSteps: int, cycles: int):
    x = 0
    env.predictv = False
    while x < cycles:
        #model = DQN.load("DQN/"+modelName, env=env, device='cuda')
        model = DQN.load("DQN/"+modelName, env=env, device='cuda')
        env.setModel(model)
        model.exploration_initial_eps = expInit
        model.exploration_final_eps = expFinal
        model.exploration_fraction = expFrac
        model._setup_model()
        model.learn(total_timesteps=totalSteps, log_interval=1)
        print('Model saved...')
        model.save("DQN/"+modelName)
        x += 1
        print("cycle start...",x)

def modelPredict(env: GameEnv, modelName: str, episodes: int):
    model = DQN.load("DQN/"+modelName, env=env)
    env.setModel(model)
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(state, deterministic=True)  # Deterministic mode for evaluation
            state, reward, done, truncated, info = env.step(action)
            #print(f"State: {state}, Predicted Action: {action}",reward)
            total_reward += reward

        print(f"Episode {episode + 1}/{episodes}: Total Reward = {total_reward}")


#Training:
#modelInit(env,logname,0.4,0.05,0.65,2000000,0.00025)
#modelInit(env,logname,0.5,0.075,0.7,1500000,0.00025)
modelPredict(env,logname,1)
##modelTrainAutomatic(env, logname, 0.05,0.025,0.7, 1000000, 2)

