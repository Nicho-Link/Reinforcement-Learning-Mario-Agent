import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_results(reward_list: list, steps_list: list, q_list:list, loss_list:list, epsilon_list:list, save_path: str):
    """Plots the results of the training

    Args:
        reward_list (list): List of rewards
        steps_list (list): List of steps
        q_list (list): List of Q values
        loss_list (list): List of losses
        epsilon_list (list): List of epsilons
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 12))

    plt.subplot(5, 1, 1) 
    plt.plot(reward_list, label="Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()

    plt.subplot(5, 1, 2)
    plt.plot(steps_list, label="Steps")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.legend()

    plt.subplot(5, 1, 3)
    plt.plot(q_list, label="Q Value")
    plt.xlabel("Episode")
    plt.ylabel("Q Value")
    plt.legend()

    plt.subplot(5, 1, 4)
    plt.plot(loss_list, label="Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(5, 1, 5)
    plt.plot(epsilon_list, label="Epsilon")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.legend()

    plt.tight_layout()
    
    # Save plot before showing
    plt.savefig(save_path)
    plt.show()

def create_plots(
    episode_rewards: list,
    episode_lengths: list,
    episode_losses: list,
    episode_epsilons: list,
    save_fig: bool = True,
    save_path: str = "results.png",
) -> None:
    """create plots during training

    Args:
        episode_rewards (list): list of rewards per episode
        episode_lengths (list): list of lengths per episode
        episode_losses (list): list of losses per episode
        episode_epsilons (list): list of epsilons per episode
        save_fig (bool, optional): decide whether to save plot. Defaults to True.
        save_path (str, optional): save path for plot. Defaults to "results.png".
    """

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
    axes = axes.flatten()

    # create plot for episode rewards
    axes[0].plot(
        np.arange(len(episode_rewards)), episode_rewards, label="Episode Rewards"
    )
    mvg_avg_reward = pd.Series(episode_rewards).rolling(5).mean().dropna()
    axes[0].plot(
        np.arange(len(mvg_avg_reward)),
        mvg_avg_reward,
        label="Moving Average",
    )
    axes[0].set_ylabel("Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_title("Episode Rewards")
    axes[0].grid(True)
    axes[0].legend()

    # create plot for episode length
    axes[1].plot(
        np.arange(len(episode_lengths)), episode_lengths, label="Episode Length"
    )
    mvg_avg_length = pd.Series(episode_lengths).rolling(5).mean().dropna()
    axes[1].plot(
        np.arange(len(mvg_avg_length)),
        mvg_avg_length,
        label="Moving Average",
    )
    axes[1].set_ylabel("Episode Length")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Episode")
    axes[1].set_title("Episode Length\n (in seconds)")
    axes[1].grid(True)
    axes[1].legend()

    # create plot for loss
    axes[2].plot(np.arange(len(episode_losses)), episode_losses, label="Loss")
    mvg_avg_loss = pd.Series(episode_losses).rolling(5).mean().dropna()
    axes[2].plot(
        np.arange(len(mvg_avg_loss)),
        mvg_avg_loss,
        label="Moving Average",
    )
    axes[2].set_ylabel("Loss")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Episode")
    axes[2].set_title("Loss per Episode")
    axes[2].grid(True)
    axes[2].legend()

    # create plot for epsilon
    axes[3].plot(np.arange(len(episode_epsilons)), episode_epsilons, label="Epsilon")
    axes[3].set_ylabel("Epsilon")
    axes[3].set_xlabel("Episode")

    axes[3].set_title("Epsilon Decay")
    axes[3].grid(True)
    axes[3].legend()

    fig.tight_layout()
    # save figure
    if save_fig:
        fig.savefig(f"{save_path}.png")