class MarioAgent:
    def __init__(self, strategy):
        """
        Initialize an Super-Mario-Bros Agent
        :param strategy: Strategy to be used by the agent (e.g. EpsilonGreedy, etc.)
        """
        self.strategy = strategy

    def selectAction(self, state):
        pass

    def saveExp(self, experience):
        pass

    def useExp(self):
        pass

    def learn(self):
        pass