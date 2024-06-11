import os, datetime
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from nes_py.wrappers import JoypadSpace
import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation, TransformObservation, Monitor
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import own Functions
from helper_functions.create_Plot import plot_results
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
buffer_size = 100000
batch_size = 32
learning_rate = 0.00025

stacking_number = 10
# skipping_number = 4

exp_before_training = 100000
online_update_every = 3
exp_before_target_sync = 10000

epsilon_start = 1.0
epsilon_min = 0.01
epsilon_decay = 0.99995
gamma = 0.99
num_episodes = 10000

save_every = 500000

vid_folder = os.path.join("videos", datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S'))

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, action_space)
env = Monitor(env, vid_folder, video_callable=lambda episode_id: True, force=True)
# env = SkipFrame(env, skip=skipping_number) # Not implemented
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = TransformObservation(env, lambda obs: np.squeeze(obs, axis=-1))
env = TransformObservation(env, f=lambda x: x / 255.)
env = FrameStack(env, num_stack=stacking_number)

state = env.reset()
state_shape = state.shape
model_folder = os.path.join("models")
checkpoint_folder = os.path.join("checkpoints", datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S'))
starting_point = None
plot_folder = os.path.join("logs", datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S'))
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)

mario = MarioAgentEpsilonGreedy(num_actions=len(action_space), state_shape=state_shape, checkpoint_folder=checkpoint_folder, model_folder=model_folder, wantcuda=True, starting_point=starting_point, learning_rate=learning_rate, epsilon_start=epsilon_start, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay, batch_size=32, gamma=gamma, buffer_size=buffer_size, exp_before_training=exp_before_training, online_update_every=online_update_every, exp_before_target_sync=exp_before_target_sync, save_every=save_every)

reward_list = []
steps_list = []
q_list = []
loss_list = []
epsilon_list = []

for episode in range(num_episodes):
    state = env.reset()
    steps = 0
    while True:
        #env.render() # Visualize
        action = mario.selectAction(state)
        steps = steps + 1
        next_state, reward, resetnow, info = env.step(action)
        mario.saveExp(state, action, next_state, reward, resetnow)
        q, loss = mario.learn_get_TDest_loss()
        state = next_state
        if resetnow or info['flag_get']:
            break
    print(f"Episode {episode + 1} abgeschlossen mit {steps} Schritten, Gesamtbelohnung: {reward}, Epsilon: {mario.epsilon}\n\n")
    
    reward_list.append(reward)
    steps_list.append(steps)
    q_list.append(q)
    loss_list.append(loss)
    epsilon_list.append(mario.epsilon)

    if episode % 50 == 0:
        plot_results(reward_list, steps_list, q_list, loss_list, epsilon_list, os.path.join(plot_folder, f"plot_{episode}.png"))
    
    if episode % save_every == 0:
        mario.saveModel(checkpoint_folder, episode=episode)
    
    mario.decayEpsilon()

mario.saveModel(model_folder)

env.close()

