import random

class ExperienceReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
 
    def add(self, state, action, next_state, reward, resetnow):
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, next_state, reward, resetnow))
        else:
            self.buffer[self.position] = (state, action, next_state, reward, resetnow)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)