import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from collections import deque


def preprocess_frame(frame: np.array) -> np.array:
    """preprocess frame by applying grayscale, resizing and normalizing

    Args:
        frame (np.array): frame of the environment

    Returns:
        np.array: preprocessed frame
    """
    # convert to grayscale
    frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    # resize the image
    frame = cv.resize(frame, (84, 84))
    # normalize the image
    frame = frame / 255.0
    return frame


def preprocess_state(
    state: np.array,
    frame_stack: deque = deque(),
    stack_size: int = 4,
) -> torch.tensor:
    """preprocessing of state consisting of apply image preprocessing and stacking frames

    Args:
        state (np.array): state of the environment
        frame_stack (deque, optional): stack of frames. Defaults to deque().
        stack_size (int, optional): number of frames to stack. Defaults to 4.

    Returns:
        torch.tensor: preprocessed and stacked state
    """
    frame = preprocess_frame(state)

    # fill frame stack with frames
    if len(frame_stack) == 0:
        for _ in range(stack_size):
            frame_stack.append(frame)
    else:
        frame_stack.append(frame)
        if len(frame_stack) > stack_size:
            frame_stack.popleft()

    state_stack = np.stack(frame_stack)

    state_stack = torch.FloatTensor(state_stack).unsqueeze(0)

    return state_stack


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
