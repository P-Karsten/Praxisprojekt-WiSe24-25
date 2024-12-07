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
from stable_baselines3.common.callbacks import BaseCallback
from pydantic import BaseModel
import httpx

learningRate = 0.0001
#learningRate = 0.00035
timesteps = 30000
#eplorationRate = 0.45
#max_steps = 500
max_steps = 500

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
        pos_LOW = np.float32(-1800)
        pos_HIGH = np.float32(1800)
        rot_LOW = np.float32(-np.pi)
        rot_HIGH = np.float32(np.pi)

        # W, A, S, D, P (shift)
        self.action_space = spaces.Discrete(2)

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
        self.state = sendAction(2)
        self.done = False

    def step(self, action):
        
        gameData = sendAction(action)
        rotation = gameData['spaceship_rotation'].item()

        # Reward
        absRotation = abs(rotation)
        self.reward = -absRotation**2

        #running rewards from active model
        if absRotation < 0.1:
            self.reward += 10

        if absRotation < 0.5:
            self.reward += 0.1

        if absRotation < 1.0:
            self.reward += 0.005

        if absRotation > np.pi / 2:
            self.reward -= 2

        """ #test
        if absRotation < 0.02:
            self.reward += 1.25

        if absRotation < 0.1:
            self.reward += 0.25

        if absRotation < 0.5:
            self.reward += 0.15

        if absRotation < 1.0:
            self.reward += 0.0085

        if absRotation > np.pi / 2:
            self.reward -= 1
        """


        global_step = getattr(self, "step_count", 0)
        with writer.as_default():
            tf.summary.scalar("reward", self.reward, step=global_step)

        self.step_count = global_step + 1


        if self.step_count >= max_steps:
            self.done = True

        self.state = gameData
        #self.done = False 
        truncated = False
        info = {}

        return self.state, self.reward, self.done, truncated, info

    
    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.done = False
        self.reward = 0
        self.state = sendAction(6)
        info = {}
        return self.state, info
    
    def render(self, mode="human"):
        ...

    def close (self):
        ...


env = GameEnv()
check_env(env)  # Check for compliance

tensorboardLogDir = "./logs/game_rewards/"

#logCallback = CustomLoggingCallback(log_dir='./logs', log_freq=10)

#model = DQN("MultiInputPolicy", env, verbose=2, tensorboard_log="./logs/game_rewards/")

#initial model learn (good exploration value -3000 to -600 in first run)
#model = DQN("MultiInputPolicy", env, verbose=2, exploration_initial_eps=1.0, exploration_final_eps=0.1, exploration_fraction=0.6, learning_rate=learningRate)
#model = DQN("MultiInputPolicy", env, verbose=2, exploration_initial_eps=1.0, exploration_final_eps=0.3, exploration_fraction=0.4, learning_rate=learningRate)

#model training with different new exploration values
model = DQN.load("dqn_spaceship", env=env)
model.set_logger(configure(tensorboardLogDir))
model.learning_rate = learningRate
 #2nd/3rd run
"""
model.exploration_initial_eps = 0.375
model.exploration_final_eps = 0.275
model.exploration_fraction = 0.5
"""

#constant train run ideal (0.1 - 0.05 later)
model.exploration_initial_eps = 0.15
#model.exploration_final_eps = 0.15
#model.exploration_fraction = 0.6


"""
#final
model.exploration_initial_eps = 0.1
model.exploration_final_eps = 0.05
model.exploration_fraction = 0.3
"""
model.learn(total_timesteps=timesteps, log_interval=5)

model.save("dqn_spaceship")


#tensorboard
#tensorboard --logdir=logs/game_rewards/