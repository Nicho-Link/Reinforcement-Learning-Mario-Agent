import os, datetime
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from nes_py.wrappers import JoypadSpace
import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation

# ???
import numpy as np
import matplotlib.pyplot as plt
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import own Functions
from helper_functions.additional_functions import FixSeedBugWrapper
from helper_functions.create_Agent import MarioAgentEpsilonGreedy



# Hyperparameters
action_space = [
    ['NOOP'],
    ['A'],
    ['B'],
    ['A', 'B'],    
    ['right'],
    ['left'],
    ['right', 'A'],
    ['left', 'A'],
    ['right', 'B'],
    ['left', 'B'],
    ['right', 'A', 'B'],
    ['left', 'A', 'B'],
    ['up'],
    ['down']
]
# learning_rate = 0.0001
discount_factor = 0.99
stacking_number = 10

epsilon_start = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

num_episodes = 10000


env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, action_space)
# env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
# env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.)
env = FrameStack(env, num_stack=4)

state = env.reset()
state_shape = state.shape

checkpoint_folder = os.path.join("checkpoints", datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S'))
starting_point = None

mario = MarioAgentEpsilonGreedy(wantcuda=True, num_actions=len(action_space), state_shape=state_shape, save_dir=checkpoint_folder, starting_point=starting_point, epsilon_start=epsilon_start, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay, batch_size=32, gamma=discount_factor)

# logger = MetricLogger(save_dir)

for episode in range(num_episodes):
    state = env.reset()
    
    while True:
        # env.render()
        action = mario.selectAction(state)
        next_state, reward, resetnow, info = env.step(action)      
        mario.saveExp(state, action, next_state, reward, resetnow)
        q, loss = mario.learn()
        # logger.log_step(reward, loss, q)
        state = next_state
        if resetnow or info['flag_get']:
            break
        
    # logger.log_episode()
    """
    if episode % 20 == 0:
        logger.record(episode, mario, epsilon)
        logger.save("logs")"""

env.close()

