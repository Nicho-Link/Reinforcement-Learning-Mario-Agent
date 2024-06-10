import torch
import numpy as np
import random
import torch.nn.functional as F

# Import own Functions
from helper_functions.create_NN import SMBAgentNN
from helper_functions.create_ExpRepBuf import ExperienceReplayBuffer

class MarioAgentEpsilonGreedy:
    def __init__(self, wantcuda, action_space, state_shape, save_dir, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=32, gamma=0.99):
        """
        Initialize an Super-Mario-Bros Agent
        :param wantcuda: Boolean, True if you want to use GPU, False if you want to use CPU
        :param action_space: List, the list of possible actions
        :param state_shape: Tuple, the shape of the state
        :param save_dir: String, the directory to save the model
        :param epsilon_start: Float, the starting value of epsilon
        :param epsilon_min: Float, the minimum value of epsilon
        :param epsilon_decay: Float, the decay rate of epsilon
        :param batch_size: Int, the batch size for the experience replay buffer
        :param gamma: Float, the discount factor gamma
        """
        if wantcuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.action_space = action_space
        self.num_actions = len(action_space) # 10

        self.state_shape = state_shape # (1, 240, 256, 10)


        self.save_dir = save_dir

        self.model = SMBAgentNN(self.state_shape, self.num_actions)
        self.model.to(self.device)

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.current_step = 0

        self.memory = ExperienceReplayBuffer()
        self.batch_size = batch_size
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def selectAction(self, state):
        """
        Choose an action based on the current state
        """
        state = torch.tensor(state, dtype=torch.float32, device=self.device)

        if np.random.rand() < self.epsilon:
            # Exploration
            action = random.choice(self.action_space)
        else:
            # Exploitation
            state = torch.tensor(state, device=self.device)
            action_values = self.model.forward(state, model="online")
            action = torch.argmax(action_values, axis=1).item()

        for i in range(len(self.action_space)):
            if self.action_space[i] == action:
                action = i
                break

        # Decay epsilon
        self.epsilon = self.epsilon * self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)

        self.current_step = self.current_step + 1

        return action


    def saveExp(self, state, action, next_state, reward, resetnow):
        state = torch.from_numpy(state)
        action = torch.tensor(action)
        next_state = torch.from_numpy(next_state)
        reward = torch.tensor(reward)
        resetnow = torch.tensor(resetnow)

        self.memory.add(state, action, next_state, reward, resetnow)

    def useExp(self):
        batch = self.memory.sample(self.batch_size)
        states, actions, next_states, rewards, resets = zip(*batch)

        states = torch.tensor(states, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        next_states = torch.tensor(next_states, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)
        resets = torch.tensor(resets, device=self.device)

        return states, actions, next_states, rewards, resets

    def estimateTDerror(self, state, action):
        current_Q = self.model.forward(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q
    
    def estimateTargetQvalues(self, next_state, reward, resets):
        with torch.no_grad():
            next_state_Q = self.model.forward(next_state, model="online")
            best_action = torch.argmax(next_state_Q, axis=1)
            next_Q = self.model(next_state, model="target")[
                np.arange(0, self.batch_size), best_action
            ]
        return (reward + (1 - resets.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def sync_Q_target(self):
        self.model.target.load_state_dict(self.model.online.state_dict())

    def train(self):
        if len(self.memory.buffer) < self.batch_size:
            return
        
        states, actions, next_states, rewards, resets = self.useExp()

        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).squeeze().to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        rewards = torch.stack(rewards).squeeze().to(self.device)
        resets = torch.stack(resets).squeeze().to(self.device)

        current_q_values = self.model.forward(states, model="online").gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.model.forward(next_states, model="online").max(1)[0]
            target_q_values = rewards + (1 - resets.float()) * self.gamma * next_q_values

        loss = F.mse_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.current_step % self.target_update == 0:
            self.sync_Q_target()

        return loss.item()