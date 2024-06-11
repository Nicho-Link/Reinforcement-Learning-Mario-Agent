import torch
import numpy as np
import torch.nn.functional as F
import os

# Import own Functions
from helper_functions.create_NN import SMBAgentNN
from helper_functions.create_ExpRepBuf import ExperienceReplayBuffer

class MarioAgentEpsilonGreedy:
    def __init__(self, num_actions, state_shape, checkpoint_folder, model_folder, wantcuda=True, starting_point=None, learning_rate=0.00025, epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.9995, batch_size=32, gamma=0.99, buffer_size=100000, exp_before_training=100000, online_update_every=3, exp_before_target_sync=10000, save_every=500000):
        """
        Initialize an Super-Mario-Bros Agent
        :param num_actions: Int, the number of possible actions
        :param state_shape: Tuple, the shape of the state
        :param checkpoint_folder: String, the directory to save the checkpoints
        :param model_folder: String, the directory to save the models
        :param wantcuda: Boolean, True if you want to use GPU, False if you want to use CPU
        :param starting_point: String, the directory of the model you want to start with
        :param learning_rate: Float, the learning rate
        :param epsilon_start: Float, the starting epsilon value
        :param epsilon_min: Float, the minimum epsilon value
        :param epsilon_decay: Float, the decay of epsilon
        :param batch_size: Int, the batch size
        :param gamma: Float, the discount factor
        :param buffer_size: Int, the size of the buffer
        :param exp_before_training: Int, the number of experiences before training
        :param online_update_every: Int, the number how often the online model gets updated
        :param exp_before_target_sync: Int, the number of experiences before updating the target model
        :param save_every: Int, the number of experiences before saving the model
        """
        if wantcuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.num_actions = num_actions
        self.state_shape = state_shape # (1, 240, 256, 10)
        self.checkpoint_folder = checkpoint_folder
        self.model_folder = model_folder
        self.starting_point = starting_point
        self.learning_rate = learning_rate
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.exp_before_training = exp_before_training
        self.online_update_every = online_update_every
        self.exp_before_target_sync = exp_before_target_sync
        self.save_every = save_every

        self.current_step = 0

        self.model = SMBAgentNN(self.state_shape, self.num_actions).float()
        self.model.to(device=self.device)

        if self.starting_point != None:
            self.load(self.starting_point)

        self.memory = ExperienceReplayBuffer(self.buffer_size)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_function = torch.nn.SmoothL1Loss()

    def selectAction(self, state):
        """
        Choose an action based on the current state
        """
        state = torch.tensor(state, dtype=torch.float32, device=self.device)

        if np.random.rand() < self.epsilon:
            # Exploration
            action = np.random.randint(self.num_actions)
        else:
            # Exploitation
            state = state.unsqueeze(0)
            action_values = self.model(state, model="online")
            action = torch.argmax(action_values, axis=1).item()

        # Decay epsilon
        self.epsilon = self.epsilon * self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)

        self.current_step = self.current_step + 1

        return action


    def saveExp(self, state, action, next_state, reward, resetnow):
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor([action]).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        resetnow = torch.FloatTensor([resetnow]).to(self.device)

        self.memory.add(state, action, next_state, reward, resetnow)

    def useExp(self):
        batch = self.memory.sample(self.batch_size)
        states, actions, next_states, rewards, resets = zip(*batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        resets = torch.stack(resets)

        return states, actions.squeeze(), next_states, rewards.squeeze(), resets.squeeze()

    def estimateTDerror(self, state, action):
        current_Q = self.model(state, model="online")[np.arange(0, self.batch_size), action]
        return current_Q
    
    def estimateQTarget(self, next_state, reward, resets):
        with torch.no_grad():
            next_state_Q = self.model(next_state, model="online")
            best_action = torch.argmax(next_state_Q, axis=1)
            next_Q = self.model(next_state, model="target")[
                np.arange(0, self.batch_size), best_action
            ]
        return (reward + (1 - resets.float()) * self.gamma * next_Q).float()

    def update_Q_online_get_loss(self, td_estimation, q_target):
        loss = self.loss_function(td_estimation, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def sync_Q_target(self):
        self.model.target.load_state_dict(self.model.online.state_dict())

    def learn_get_TDest_loss(self):
        if self.current_step % self.exp_before_target_sync == 0:
            self.sync_Q_target()

        if self.current_step % self.save_every == 0:
            self.saveModel(self.checkpoint_folder)

        if self.current_step < self.exp_before_training:
            return None, None
        
        if self.current_step % self.online_update_every != 0:
            return None, None
        
        states, actions, next_states, rewards, resets = self.useExp()

        td_estimation = self.estimateTDerror(states, actions)

        q_target = self.estimateQTarget(next_states, rewards, resets)

        loss = self.update_Q_online_get_loss(td_estimation, q_target)

        return (td_estimation.mean().item(), loss)
    
    def saveModel(self, path, episode=None):
        if episode == None:
            save_dir = os.path.join(path, f"Checkpoint_{int(self.current_step // self.save_every)}.chkpt")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(dict(model=self.model.state_dict(), epsilon=self.epsilon), save_dir)
            print(f"Step: {self.current_step}\nModel saved at {save_dir}")
        else:
            save_dir = os.path.join(path, f"Checkpoint_{episode}.chkpt")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(dict(model=self.model.state_dict(), epsilon=self.epsilon), save_dir)
            print(f"Step: {self.current_step}\nModel saved at {save_dir}")

    def loadModel(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        epsilon = checkpoint["epsilon"]
        model = checkpoint["model"]

        print(f"Model loaded from {path} with epsilon {epsilon}")
        self.model.load_state_dict(model)
        self.epsilon = epsilon
