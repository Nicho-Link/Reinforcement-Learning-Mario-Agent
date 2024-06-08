import numpy as np
from tqdm import tqdm

def decayingEpsilonGeedy(env, init_epsilon=1.0, min_epsilon=0.01, decay_ratio=0.1, n_episodes=1000):
    
    Q = np.zeros((env.action_space.n), dtype=np.float64)
    N = np.zeros((env.action_space.n), dtype=np.int)

    Qe = np.empty((n_episodes, env.action_space.n), dtype=np.float64)
    returns = np.empty(n_episodes, dtype=np.float64)
    actions = np.empty(n_episodes, dtype=np.int)

    decay_episodes = int(n_episodes * decay_ratio)
    rem_episodes = n_episodes - decay_episodes
    epsilons = 0.01
    epsilons /= np.logspace(-2, 0, decay_episodes)
    epsilons *= init_epsilon - min_epsilon
    epsilons += min_epsilon
    epsilons = np.pad(epsilons, (0, rem_episodes), 'edge')
    name = 'Exp Epsilon-Greedy {}, {}, {}'.format(init_epsilon, 
                                          min_epsilon, 
                                          decay_ratio)
    for e in tqdm(range(n_episodes), 
                  desc='Episodes for: ' + name, 
                  leave=False):
        if np.random.uniform() > epsilons[e]:
            action = np.argmax(Q)
        else:
            action = np.random.randint(len(Q))

        _, reward, _, _ = env.step(action)
        N[action] += 1
        Q[action] = Q[action] + (reward - Q[action])/N[action]
        
        Qe[e] = Q
        returns[e] = reward
        actions[e] = action
    return name, returns, Qe, actions