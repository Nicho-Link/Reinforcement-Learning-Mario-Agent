import gym

class FixSeedBugWrapper(gym.Wrapper):
    """
    Custom Wrapper to fix an Bug that appears when reseting the environment
    """
    def reset(self, **kwargs):
        # Remove seed and options arguments if they exist
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        obs = self.env.reset(**kwargs)
        return obs