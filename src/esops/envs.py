from dataclasses import dataclass
from dataclasses import field

import numpy as np

from esops.utils import temporary_seed


@dataclass
class BaseEnv:
    seed: int
    num_items: int
    time_span: int
    history_span: int
    low: int = field(init=True, default=0)
    high: int = field(init=True, default=1000)
    scale_factor: int = field(init=True, default=5)

    def __post_init__(self):
        with temporary_seed(self.seed):
            self.H = self._generate_env().astype(int)
            self.M = self.H[:, -self.time_span:]  # views

    def _generate_env(self):
        init = np.random.randint(self.low, self.high, size=self.num_items)  # generate random steps for all items and ts
        scale = (self.high - self.low) / self.scale_factor
        random_steps = np.random.normal(loc=0, scale=scale, size=(self.num_items, self.history_span))
        matrix = np.cumsum(random_steps, axis=1)  # create the cumulative sum for the random walk
        matrix += init[:, None]  # add the initial views to the first column
        matrix = np.maximum(matrix, 0)  # make sure no negatives
        return matrix

    def save_matrix(self, path):
        with open(path, "wb") as f:
            np.save(f, self.M)

    @staticmethod
    def load_matrix(path):
        with open(path, "rb") as f:
            return np.load(f)


class ConvertedRewardsEnv(BaseEnv):

    def __init__(self, seed, num_items, time_span, history_span, low: int = 0, high: int = 1000, scale_factor: int = 5):
        super().__init__(seed, num_items, time_span, history_span, low, high, scale_factor)
        self.cr = self._generate_conversion_rates(low=0.005, high=0.02, random_scale=0.0005)
        self.R = (self.M * self.cr).astype(int)  # converted views

    def _generate_conversion_rates(self, low=0.05, high=0.2, random_scale=0.5):
        """
        Simulate conversion rates with random walk.

        Parameters:
        - num_items (int): Number of items.
        - num_timesteps (int): Number of timesteps.
        - low (float): Minimum conversion rate.
        - high (float): Maximum conversion rate.
        - random_scale (float): Scale of random changes at each timestep.

        Returns:
        - numpy.ndarray: Simulated conversion rates for each item over time.
        """
        init_rates = np.random.uniform(low, high, size=self.num_items)
        random_steps = np.random.normal(loc=0, scale=random_scale, size=(self.num_items, self.time_span))
        conversion_rates = np.cumsum(random_steps, axis=1)
        conversion_rates += init_rates[:, None]
        conversion_rates = np.clip(conversion_rates, low, high)
        return conversion_rates
