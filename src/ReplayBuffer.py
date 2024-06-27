import collections
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        """
        Args:
            capacity (int): maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=self.capacity)

        self.size = 0

    def push(self, experience: collections.namedtuple) -> None:
        """save experience in buffer

        Args:
            experience (collections.namedtuple): tuple containing state, action, reward, done, next_state
        """
        self.buffer.append(experience)

        self.size += 1
        self.size = min(self.size, self.capacity)

    def sample(self, batch_size: int) -> tuple:
        """select random sample from buffer

        Args:
            batch_size (int): number of samples to select

        Returns:
            tuple: states, actions, rewards, dones, next_states
        """
        indices = np.random.choice(self.size, batch_size)
        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices]
        )
        return (
            states,
            actions,
            rewards,
            dones,
            next_states,
        )
