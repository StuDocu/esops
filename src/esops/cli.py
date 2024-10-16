import os
import json
import uuid

import click
import numpy as np

from esops.agents.ql import SMQLearningAgent, NonAgent
from esops.envs import ConvertedRewardsEnv
from esops.utils import make_timeline_plot


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
        self.id = None
        self.path = None
        if name is not None:
            self.setup(name, seed)

    def setup(self, name, seed):
        self.seed = seed
        self.name = name
        self.id: str = uuid.uuid4().hex[:16]
        self.path = f"./experiments/seed={self.seed}/{self.name}/run={self.id}"
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
    context.obj = xp


@experiment.command()
@click.option("--num-items", type=int, default=20)
@click.option("--time-span", type=int, default=15)
@click.option("--history-span", type=int, default=3000)
@click.option("--low", type=int, default=0)
@click.option("--high", type=int, default=10)
@click.option("--alpha", type=float, default=0.1)
@click.option("--gamma", type=float, default=0.9)
@click.option("--epsilon", type=float, default=0.1)
@click.pass_obj
def run_q_learning(context, num_items, time_span, low, high, history_span, alpha, gamma, epsilon):
    """
    poetry run simulate --seed 128 run-q-learning --alpha 0.01 --gamma 0.5 --epsilon 0.1
    """
    click.echo(F"seed-id: seed={context.seed}")
    click.echo(F"run-id: run={context.id}")
    env = ConvertedRewardsEnv(
        seed=context.seed,
        num_items=num_items,
        time_span=time_span,
        history_span=history_span,
        low=low,
        high=high,
    )
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


@experiment.command()
@click.option("--num-runs", type=int, default=10)
@click.option("--num-items", type=int, default=20)
@click.option("--time-span", type=int, default=15)
@click.option("--history-span", type=int, default=3000)
@click.option("--low", type=int, default=0)
@click.option("--high", type=int, default=10)
@click.option("--alpha", type=float, default=0.1)
@click.option("--gamma", type=float, default=0.9)
@click.option("--epsilon", type=float, default=0.1)
@click.pass_obj
@click.pass_context
def run_simulation(ctx, context, num_runs, num_items, time_span, low, high, history_span, alpha, gamma, epsilon):
    """
    poetry run experiment --seed 48 run-simulation
    """
    name = f"n={num_items}-t={time_span}-h={history_span}-a={alpha}-g={gamma}-e={epsilon}-low={low}-high={high}"
    for i in range(num_runs):
        context.setup(seed=context.seed, name=name)
        ctx.invoke(
            run_q_learning,
            num_items=num_items,
            time_span=time_span,
            low=low,
            high=high,
            history_span=history_span,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
        )


if __name__ == "__main__":
    experiment()
