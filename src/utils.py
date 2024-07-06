from collections import deque

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch


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


def plot_results(
    reward_list: list,
    steps_list: list,
    q_list: list,
    loss_list: list,
    epsilon_list: list,
    save_fig: bool = True,
    save_path: str = "results.png",
):
    """Plots the results of the training

    Args:
        reward_list (list): List of rewards
        steps_list (list): List of steps
        q_list (list): List of Q values
        loss_list (list): List of losses
        epsilon_list (list): List of epsilons
        save_fig (bool): Flag to save the plot
        save_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 12))
    axes = axes.flatten()

    axes[0].plot(reward_list, label="Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].legend()

    axes[1].plot(steps_list, label="Steps")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")
    axes[1].legend()

    axes[2].plot(q_list, label="Q Value")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Q Value")
    axes[2].legend()

    axes[3].plot(loss_list, label="Loss")
    axes[3].set_xlabel("Episode")
    axes[3].set_ylabel("Loss")
    axes[3].legend()

    axes[4].plot(epsilon_list, label="Epsilon")
    axes[4].set_xlabel("Episode")
    axes[4].set_ylabel("Epsilon")
    axes[4].legend()

    fig.tight_layout()

    # Save plot before showing
    if save_fig:
        fig.savefig(save_path)
