import random

class ExperienceReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
 
    def add(self, state, action, next_state, reward, resetnow):
        self.buffer[self.position] = (state, action, reward, next_state, resetnow)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)