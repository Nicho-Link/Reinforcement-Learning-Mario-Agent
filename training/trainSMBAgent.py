from nes_py.wrappers import JoypadSpace
import gym
import gym_super_mario_bros
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import cv2
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, action_space)
# env = SkipFrame(env, skip=4)
# env = GrayScaleObservation(env, keep_dim=True)
# env = ResizeObservation(env, shape=84)
# env = FrameStack(env, num_stack=4)

env.reset()
next_state, reward, done, terminated, info = env.step(action=0)
print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

done = True
env.reset()
for act_step in range(5000):
    action = env.action_space.sample()
    next_state, reward, done, terminated, info = env.step(action)
    if done:
       env.reset()

env.close()



state_size = env.observation_space.shape
action_size = env.action_space.n

agent = DDQNAgent(state_size, action_space, buffer_size=10000, batch_size=64, gamma=0.99, learning_rate=0.0001, tau=0.001, update_every=4)

n_episodes = 1000
max_t = 10000
eps_start = 1.0
eps_end = 0.01
eps_decay = 0.995

for i_episode in range(1, n_episodes+1):
    state = env.reset()
    state = np.array(state)
    score = 0
    eps = max(eps_end, eps_decay*eps_start)
    
    for t in range(max_t):
        action = agent.act(state, eps)
        next_state, reward, done, _ = env.step(action)
        next_state = np.array(next_state)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break
    
    print(f"Episode {i_episode}/{n_episodes}, Score: {score}")

env.close()


