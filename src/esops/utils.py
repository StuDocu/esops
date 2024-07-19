import contextlib
import numpy as np
from matplotlib import pyplot as plt


@contextlib.contextmanager
def temporary_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def make_timeline_plot(rewards, choices):
    num_items, time_steps = rewards.shape

    plt.figure(figsize=(32, 16))
    colors = plt.cm.get_cmap('tab10', num_items)  # Use a colormap for different lines

    for i in range(num_items):
        plt.plot(
            range(time_steps),
            rewards[i, :],
            label=f'Rewards Item {i}',
            marker='o',
            linestyle='-',
            markersize=5,
            color=colors(i)
        )

    for t, choice in enumerate(choices):
        for c in choice:
            plt.scatter(
                t,
                rewards[c, t],
                color='red',
                s=100,
                edgecolors='black',
                zorder=5,
                label=f'Selected Item {c}' if t == 0 else None,
            )

    plt.xlabel('Timesteps', fontsize=14)
    plt.ylabel('Views', fontsize=14)
    plt.title('Rewards and Selections over Time', fontsize=16)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    return plt
