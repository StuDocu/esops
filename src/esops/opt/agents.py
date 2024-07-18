from abc import ABC, abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Union

import numpy as np


@dataclass
class BaseAgent(ABC):
    ts: int
    num_items: int
    hist: list = field(default_factory=list)
    choices: list = field(default_factory=list)

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def step(self, *args, **kwargs):
        pass

    @abstractmethod
    def selection_policy(self, *args, **kwargs):
        pass

    @abstractmethod
    def update_policy(self, *args, **kwargs):
        pass

    @staticmethod
    def softmax(x: Union[list, np.ndarray]):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    @staticmethod
    def log_taylor_softmax(x: Union[list, np.ndarray], order: int = 2):
        """
        Compute the Taylor series expansion of the log-softmax function.
        """
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        log_approx = exp_x - 1 + (exp_x - 1) ** 2 / 2 if order == 2 else exp_x - 1  # add higher-order terms as needed
        softmax_probs = log_approx / np.sum(log_approx)
        return softmax_probs


class NonAgent(BaseAgent):
    name = 'existing-approach'

    def __init__(self, ts, num_items, history):
        super().__init__(ts, num_items)
        self.history = history
        arr = np.sum(history, axis=1)
        sorted_indices = np.argsort(arr, )[::-1]
        sorted_arr = arr[sorted_indices]
        total_sum = np.sum(sorted_arr)
        threshold = 0.6 * total_sum
        cum_sum = np.cumsum(sorted_arr)
        choices = []
        for i in range(5):
            if cum_sum[i] < threshold:
                choices.append(int(i))
            else:
                choices.append(int(i))
                break

        self.choice = sorted_indices[choices].tolist()  # static

    def step(self, rewards, n):
        self.hist.append(rewards)
        self.choices.append(self.choice)
        return self.choice

    def selection_policy(self, *args, **kwargs):
        pass

    def update_policy(self, *args, **kwargs):
        pass


class GreedyRandomizedRLAgent(BaseAgent):
    """
    Naive Reinforcement Learning Agent, that uses softmax selection policy.
    """

    name = "greedy-rl"

    def __init__(self,
                 ts: int,
                 num_items: int,
                 alpha: float,
                 init_probas: Union[list, np.ndarray] = None,
                 entropy: float = 0.01
                 ):
        super().__init__(ts, num_items)
        self.alpha = alpha
        self.entropy = entropy
        self.updated_probas = []
        self.probas = self._set_init_probas(init_probas)
        self.updated_probas.append(self.probas.copy())

    def _set_init_probas(self, init_probas):
        if init_probas is None:
            return np.ones(self.num_items) / self.num_items
        return init_probas

    def step(self, rewards: Union[list, np.ndarray], n: int):
        """
        Selects n items based on the rewards and updates the policy.
        Parameters
        ----------
        rewards: list or np.ndarray
        n: int
        """
        self.hist.append(rewards)
        self.update_policy(rewards, n)
        choices = self.selection_policy(self.probas, n)
        self.choices.append(choices.tolist())
        return self.choices[-1]

    def selection_policy(self, probas, n):
        return np.random.choice(len(probas), n, replace=False, p=probas)

    def update_policy(self, rewards: Union[list, np.ndarray], n: int):
        softmax_probas = self.softmax(rewards)
        probas = (1 - self.alpha) * self.probas + self.alpha * softmax_probas
        entropy = -np.sum(probas * np.log(probas + 1e-12))
        probas = probas + self.entropy * entropy / self.num_items
        probas = probas / np.sum(probas)  # Normalize to ensure they sum to 1
        self.probas = probas
        self.updated_probas.append(probas.copy())


class ExtGreedyRLAgent(GreedyRandomizedRLAgent):
    """
    Extrapolated Reinforcement Learning Agent, that uses softmax selection policy.
    """

    name = "ext-greedy-rl"

    def __init__(self,
                 ts: int,
                 num_items: int,
                 alpha: float,
                 init_probas: Union[list, np.ndarray] = None,
                 entropy: float = 0.01
                 ):
        super().__init__(ts, num_items, alpha, init_probas, entropy)

    def step(self, rewards: Union[list, np.ndarray], n: int):
        """
        Selects n items based on the rewards and updates the policy.
        Parameters
        ----------
        rewards: list or np.ndarray
        n: int
        """
        self.hist.append(rewards)
        if len(self.hist) < 5:
            ext_rewards = rewards
        else:
            ext_rewards = self.extrapolate()
        self.update_policy(ext_rewards, n)
        choices = self.selection_policy(self.probas, n)
        self.choices.append(choices.tolist())
        return self.choices[-1]

    def extrapolate(self):
        """
        Predict future rewards using autoregressive model.
        """
        _hist = np.asarray(self.hist).T
        phi = np.array([np.corrcoef(_hist[i, :-1], _hist[i, 1:])[0, 1] for i in range(_hist.shape[0])])
        means = np.mean(_hist, axis=1)
        last_values = _hist[:, -1]
        extrapolations = means + phi * (last_values - means)
        return extrapolations


class ExtEpsilonRLAgent(BaseAgent):
    name = 'ext-epsilon-rl'

    def __init__(self,
                 ts: int,
                 num_items: int,
                 init_probas: Union[list, np.ndarray] = None,
                 alpha: float = 0.1
                 ):
        super().__init__(ts, num_items)
        self.alpha = alpha
        self.updated_probas = []
        self.probas = self._set_init_probas(init_probas)
        self.updated_probas.append(self.probas.copy())

    def _set_init_probas(self, init_probas):
        if init_probas is None:
            return np.ones(self.num_items) / self.num_items
        return init_probas

    def step(self, rewards: Union[list, np.ndarray], n: int):
        """
        Selects n items based on the rewards and updates the policy.
        Parameters
        ----------
        rewards: list or np.ndarray
        n: int
        """
        self.hist.append(rewards)
        if len(self.hist) < 5:
            ext_rewards = rewards
        else:
            ext_rewards = self.extrapolate()
        self.update_policy(ext_rewards)
        choices = self.selection_policy(self.probas, n)
        self.choices.append(choices.tolist())
        return self.choices[-1]

    def selection_policy(self, probas, n):
        return np.random.choice(len(probas), n, replace=False, p=probas)

    def update_policy(self, rewards: Union[list, np.ndarray]):
        reward_probas = rewards / rewards.sum()
        probas = self.probas + self.alpha * (reward_probas - self.probas)
        self.probas = probas
        self.updated_probas.append(probas.copy())

    def extrapolate(self):
        """
        Predict future rewards using autoregressive model.
        """
        _hist = np.asarray(self.hist).T
        phi = np.array([np.corrcoef(_hist[i, :-1], _hist[i, 1:])[0, 1] for i in range(_hist.shape[0])])
        phi[np.isnan(phi)] = 0
        means = np.mean(_hist, axis=1)
        last_values = _hist[:, -1]
        extrapolations = means + phi * (last_values - means)
        return extrapolations


class QLearningAgent(BaseAgent, ABC):
    name = 'q-learning-rl'

    def __init__(self,
                 ts: int,
                 num_items: int,
                 alpha: float = 0.1,
                 gamma: float = 0.9,
                 epsilon: float = 0.1
                 ):
        super().__init__(ts, num_items)
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.updated_q_values = []
        self.q_values = np.zeros(num_items)  # initialize q-values

    def step(self, rewards: Union[list, np.ndarray], n: int):
        """
        Selects n items based on the rewards and updates the Q-values.
        Parameters
        ----------
        rewards: list or np.ndarray
        n: int
        """
        if len(self.hist) > 0:
            self.update_policy(rewards)
        self.hist.append(rewards)
        choices = self.selection_policy(n)
        self.choices.append(choices.tolist())
        return self.choices[-1]

    @abstractmethod
    def selection_policy(self, *args, **kwargs):
        pass

    def _update_q_values(self, rewards: Union[list, np.ndarray], choices: list):
        """
        Update Q-values based on the received rewards
        """
        for idx in choices:
            # update rule: Q(s) = Q(s) + alpha * (reward + gamma * max(Q(s')) - Q(s))
            self.q_values[idx] += self.alpha * (rewards[idx] - self.q_values[idx])

    def update_policy(self, rewards: Union[list, np.ndarray]):
        self._update_q_values(rewards, self.choices[-1])
        self.updated_q_values.append(self.q_values.copy())


class EpsilonGreedyQLearningAgent(QLearningAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def selection_policy(self, n):
        """
        Epsilon-greedy policy for selecting items
        """
        if np.random.rand() < self.epsilon:
            # Exploration: randomly select items
            return np.random.choice(self.num_items, n, replace=False)
        else:
            # Exploitation: select items with the highest Q-values
            return np.argsort(self.q_values)[-n:]


class SoftQLearningAgent(QLearningAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def selection_policy(self, n):
        """
        Softmax selection policy for selecting items
        """
        exp_q_values = self.softmax(self.q_values)
        probas = exp_q_values / exp_q_values.sum()
        return np.random.choice(len(self.q_values), n, replace=False, p=probas)
