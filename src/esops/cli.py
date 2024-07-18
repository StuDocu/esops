import os
import json
import uuid

import click
import numpy as np
from matplotlib import pyplot as plt

from esops.opt.agents import SoftQLearningAgent, NonAgent
from esops.envs import ConvertedRewardsEnv


np.random.seed(0)


class ExperimentContext:
    def __init__(self, name: str = None):
        if name is None:
            self.name: str = uuid.uuid4().hex[:16]
        else:
            self.name = name

    def setup(self):
        os.makedirs(f"./experiments/{self.name}", exist_ok=True)

    def add_config(self, config: dict):
        with open(f"experiments/{self.name}/config.json", "w") as f:
            json.dump(config, f, indent=4)

    def add_plot(self, plot, to="plot.png"):
        plot.savefig(f"experiments/{self.name}/{to}")

    def add_arrays(self, agent):
        with open(f"experiments/{self.name}/choices.json", "w") as f:
            json.dump({i: agent.choices[i] for i in range(len(agent.choices))}, f, indent=4)
        with open(f"experiments/{self.name}/rewards.json", "w") as f:
            hist = np.asarray(agent.hist).tolist()
            json.dump({i: hist[i] for i in range(len(hist))}, f, indent=4)
        with open(f"experiments/{self.name}/probas.json", "w") as f:
            if hasattr(agent, 'updated_probas'):
                probas = np.asarray(agent.updated_probas).tolist()
                json.dump({i: probas[i] for i in range(len(probas))}, f, indent=4)
            elif hasattr(agent, 'updated_q_values'):
                q_values = np.asarray(agent.updated_q_values).tolist()
                json.dump({i: q_values[i] for i in range(len(q_values))}, f, indent=4)

    def add_sum_reward(self, env, agent):
        sum_reward = int(env.R[agent.choices].sum())
        sum_views = int(env.M[agent.choices].sum())
        with open(f"experiments/{self.name}/sum_rewards.json", "w") as f:
            json.dump({'R': sum_reward, 'V': sum_views}, f, indent=4)


@click.group()
@click.option("--name", type=str)
@click.pass_context
def experiment(context, name):
    xp = ExperimentContext(name=name)
    xp.setup()
    context.obj = xp


@experiment.command()
@click.option("--num-items", type=int, default=5)
@click.option("--num-choices", type=int, default=2)
@click.option("--time-steps", type=int, default=10)
@click.option("--alpha", type=float, default=0.1)
@click.option("--gamma", type=float, default=0.9)
@click.option("--epsilon", type=float, default=0.1)
@click.pass_obj
def run_q_learning(context, num_items, num_choices, time_steps, alpha, gamma, epsilon):
    """
    poetry run simulate --name 'rl-experiment-baseline' run --alpha 0.1 --num-items 8 --num-choices=2 --time-steps 50
    """
    click.echo(F"Experiment ID: {context.name}")
    env = ConvertedRewardsEnv(num_items=num_items, ts=time_steps)
    agent = SoftQLearningAgent(
        ts=time_steps,
        num_items=num_items,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
    )
    # control_agent = NonAgent(
    #     history=env.H[:, :-time_steps],
    #     num_items=num_items, ts=time_steps)

    context.add_config(
        {
            'agent': agent.name,
            'num_items': num_items,
            'num_choices': num_choices,
            'time_steps': time_steps,
            'alpha': alpha,
            'gamma': gamma,
            'epsilon': epsilon,
        }
    )

    for t in range(1, time_steps):
        agent.step(env.R[:, t], num_choices)

    context.add_arrays(agent)
    context.add_sum_reward(env, agent)

    agent_view_plot = make_timeline_plot(env.M, agent.choices)
    context.add_plot(agent_view_plot, 'view.plot.png')

    agent_reward_plot = make_timeline_plot(env.R, agent.choices)
    context.add_plot(agent_reward_plot, 'rewards.plot.png')


# @experiment.command()
# @click.option('--course-id', type=int, default=4527748)
# @click.option('--seed', type=int, default=12)
# @click.option("--num-choices", type=int, default=2)
# @click.option("--alpha", type=float, default=0.1)
# @click.option("--gamma", type=float, default=0.9)
# @click.option("--epsilon", type=float, default=0.1)
# @click.pass_obj
# def run_q_learning_init(context, course_id, seed, num_choices, alpha, gamma, epsilon):
#     """
#     poetry run simulate --name 'rl-experiment-baseline' run --alpha 0.1 --num-items 8 --num-choices=2 --time-steps 50
#     """
#     click.echo(F"Experiment ID: {context.name}")
#     env = RealViewsConvertedRewardsEnv(course_id=course_id, seed=seed)
#     agent = SoftQLearningAgent(
#         ts=env.ts,
#         num_items=env.num_items,
#         alpha=alpha,
#         gamma=gamma,
#         epsilon=epsilon,
#     )
#
#     context.add_config(
#         {
#             'agent': agent.name,
#             'num_items': env.num_items,
#             'num_choices': num_choices,
#             'time_steps': env.ts,
#             'alpha': alpha,
#             'gamma': gamma,
#             'epsilon': epsilon,
#         }
#     )
#
#     for t in range(1, env.ts):
#         agent.step(env.R[:, t], num_choices)
#     context.add_arrays(agent)
#     context.add_sum_reward(env, agent)
#
#     plot = make_timeline_plot(env.M, agent.choices)
#     context.add_plot(plot, 'view.plot.png')
#
#     plot = make_timeline_plot(env.R, agent.choices)
#     context.add_plot(plot, 'rewards.plot.png')


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


if __name__ == "__main__":
    experiment()
