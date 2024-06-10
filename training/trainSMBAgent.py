from nes_py.wrappers import JoypadSpace
import gym
import gym_super_mario_bros
import torch
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import numpy as np
import matplotlib.pyplot as plt

# Import own Functions
from helper_functions.additional_functions import FixSeedBugWrapper
from helper_functions.create_Agent import MarioAgentEpsilonGreedy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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




env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, action_space)
env = FixSeedBugWrapper(env)
env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=stacking_number, channels_order="last")

state_shape = (1, 240, 256, stacking_number)



mario = MarioAgentEpsilonGreedy(wantcuda=True, action_space=action_space, state_shape=state_shape, save_dir="models/", epsilon_start=epsilon_start, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay, batch_size=32, gamma=discount_factor)






for episode in range(num_episodes):
    state = env.reset()
    state = np.array(state)
    score = 0
    
    while True:
        action = mario.selectAction(state)
        next_state, reward, resetnow, info = env.step([action])
        next_state = np.array(next_state)
        
        mario.saveExp(state, action, next_state, reward, resetnow)
        mario.train()
        
        state = next_state
        score += reward
        
        if resetnow:
            break
    
    print(f"Episode {episode}/{num_episodes}, Score: {score}")

env.close()

