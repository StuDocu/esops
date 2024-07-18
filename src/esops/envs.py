from dataclasses import dataclass
from dataclasses import field

import numpy as np


np.random.seed(12)


@dataclass
class BaseEnv:
    num_items: int
    ts: int
    low: int = field(init=True, default=0)
    high: int = field(init=True, default=1000)

    def __post_init__(self):
        self.M = self._generate_env()

    def _generate_env(self):
        init = np.random.randint(1, 1000, size=self.num_items)  # generate random steps for all items and ts
        random_steps = np.random.normal(loc=0, scale=50, size=(self.num_items, self.ts))
        matrix = np.cumsum(random_steps, axis=1)  # create the cumulative sum for the random walk
        matrix += init[:, None]  # add the initial views to the first column
        matrix = np.maximum(matrix, 0)  # make sure no negatives
        return matrix


class ConvertedRewardsEnv(BaseEnv):

    def __init__(self, num_items, ts, low: int = 0, high: int = 1000):
        super().__init__(num_items, ts, low, high)
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
        random_steps = np.random.normal(loc=0, scale=random_scale, size=(self.num_items, self.ts))
        conversion_rates = np.cumsum(random_steps, axis=1)
        conversion_rates += init_rates[:, None]
        conversion_rates = np.clip(conversion_rates, low, high)
        return conversion_rates
