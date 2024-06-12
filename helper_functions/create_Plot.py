import matplotlib.pyplot as plt

def plot_results(reward_list, steps_list, q_list, loss_list, epsilon_list, save_path):
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