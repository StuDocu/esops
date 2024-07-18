import os
import json
import uuid

import click
import numpy as np
from matplotlib import pyplot as plt

from esops.opt.agents import SMQLearningAgent, NonAgent
from esops.envs import ConvertedRewardsEnv


np.seterr(divide='ignore')


def compute_total_reward(env, agent):
    r = [env.R.T[i][chs].sum() for i, chs in enumerate(agent.choices)]
    return int(sum(r))


def compute_total_views(env, agent):
    r = [env.M.T[i][chs].sum() for i, chs in enumerate(agent.choices)]
    return int(sum(r))


class ExperimentContext:
    def __init__(self, seed, name: str = None):
        self.name = name
        self.seed = seed
        if name is None:
            self.id: str = uuid.uuid4().hex[:16]
        else:
            self.id = name
        self.path = f"./experiments/seed={self.seed}/{self.name}/run={self.id}"

    def setup(self):
        os.makedirs(self.path, exist_ok=True)

    def add_config(self, config: dict):
        with open(f"{self.path}/config.json", "w") as f:
            json.dump(config, f, indent=4)

    def add_plot(self, plot, to="plot.png"):
        plot.savefig(f"{self.path}/{to}")

    def add_arrays(self, agent, control_agent):
        with open(f"{self.path}/choices.json", "w") as f:
            json.dump({i: agent.choices[i] for i in range(len(agent.choices))}, f, indent=4)
        with open(f"{self.path}/control.choices.json", "w") as f:
            json.dump(control_agent.choice, f, indent=4)
        with open(f"{self.path}/rewards.json", "w") as f:
            hist = np.asarray(agent.hist).tolist()
            json.dump({i: hist[i] for i in range(len(hist))}, f, indent=4)
        with open(f"{self.path}/qvalues.json", "w") as f:
            q_values = np.asarray(agent.updated_q_values).tolist()
            json.dump({i: q_values[i] for i in range(len(q_values))}, f, indent=4)

    def add_sum_reward(self, env, agent, control_agent):
        sum_reward = compute_total_reward(env, agent)
        sum_views = compute_total_views(env, agent)
        control_reward = compute_total_reward(env, control_agent)
        control_views = compute_total_views(env, control_agent)
        with open(f"{self.path}/sum_rewards.json", "w") as f:
            json.dump(
                {
                    'agent': {'R': sum_reward, 'V': sum_views},
                    'control': {'R': control_reward, 'V': control_views},
                },
                f,
                indent=4
            )


@click.group()
@click.option("--seed", type=int)
@click.option("--name", type=str)
@click.pass_context
def experiment(context, seed, name):
    xp = ExperimentContext(seed=seed, name=name)
    xp.setup()
    context.obj = xp


@experiment.command()
@click.option("--num-items", type=int, default=20)
@click.option("--time-span", type=int, default=100)
@click.option("--history-span", type=int, default=3000)
@click.option("--alpha", type=float, default=0.1)
@click.option("--gamma", type=float, default=0.9)
@click.option("--epsilon", type=float, default=0.1)
@click.pass_obj
def run_q_learning(context, num_items, time_span, history_span, alpha, gamma, epsilon):
    """
    poetry run simulate --seed 128 run-q-learning --alpha 0.01 --gamma 0.5 --epsilon 0.1

    Parameters
    ----------
    context: ExperimentContext
    num_items: int
    time_span: int
    history_span: int
    alpha: float
    gamma: float
    epsilon: float
    """
    click.echo(F"seed-id: seed={context.seed}")
    click.echo(F"run-id: run={context.id}")
    env = ConvertedRewardsEnv(seed=context.seed, num_items=num_items, time_span=time_span, history_span=history_span)
    control_agent = NonAgent(num_items=num_items, ts=time_span, history=env.H[:, :-time_span])
    num_choices = len(control_agent.choice)

    init_q_values = np.zeros(num_items)
    for idx in control_agent.choice:
        init_q_values[idx] += 5

    agent = SMQLearningAgent(
        ts=time_span,
        num_items=num_items,
        init_q_values=init_q_values,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
    )

    context.add_config(
        {
            'agent': agent.name,
            'num_items': num_items,
            'num_choices': num_choices,
            'time_span': time_span,
            'history_span': env.history_span,
            'alpha': alpha,
            'gamma': gamma,
            'epsilon': epsilon,
        }
    )

    for t in range(time_span):
        agent.step(env.R[:, t], num_choices)
        control_agent.step(env.R[:, t], num_choices)

    context.add_arrays(agent, control_agent)
    context.add_sum_reward(env, agent, control_agent)

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
