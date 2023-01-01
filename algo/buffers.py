import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union, NamedTuple

import numpy as np
import torch as th

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor

class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    reward_estimates: th.Tensor
    
class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_dim: Observation dim
    :param action_dim: Action dim
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_dim: int,
        action_dim: int,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.pos = 0
        self.full = False
        self.device = device
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)

    @abstractmethod
    def _get_samples(
        self, batch_inds: np.ndarray
    ) -> Union[ReplayBufferSamples, RolloutBufferSamples]:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return:
        """
        if copy:
            return th.tensor(array).to(self.device)
        return th.as_tensor(array).to(self.device)

    @staticmethod
    def _normalize_obs(
        obs: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:

        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray) -> np.ndarray:

        return reward


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_dim: Observation dim
    :param action_dim: Action dim
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_dim: int,
        action_dim: int,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_dim, action_dim, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs, self.observation_dim), dtype=np.float32)

        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, self.observation_dim), dtype=np.float32)

        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.reward_estimates = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.reward_estimates.nbytes + self.dones.nbytes

            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        reward_estimate: np.ndarray,
        done: np.ndarray
    ) -> None:

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.reward_estimates[self.pos] = np.array(reward_estimate).copy()
        self.dones[self.pos] = np.array(done).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :])
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :])

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :]),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1)),
            self._normalize_reward(self.reward_estimates[batch_inds, env_indices].reshape(-1, 1))
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


# class RolloutBuffer(BaseBuffer):
#     """
#     Rollout buffer used in on-policy algorithms like A2C/PPO.
#     It corresponds to ``buffer_size`` transitions collected
#     using the current policy.
#     This experience will be discarded after the policy update.
#     In order to use PPO objective, we also store the current value of each state
#     and the log probability of each taken action.

#     The term rollout here refers to the model-free notion and should not
#     be used with the concept of rollout used in model-based RL or planning.
#     Hence, it is only involved in policy and value function training but not action selection.

#     :param buffer_size: Max number of element in the buffer
#     :param observation_dim: Observation dim
#     :param action_dim: Action dim
#     :param device:
#     :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
#         Equivalent to classic advantage when set to 1.
#     :param gamma: Discount factor
#     :param n_envs: Number of parallel environments
#     """

#     def __init__(
#         self,
#         buffer_size: int,
#         observation_dim: int,
#         action_dim: int,
#         device: Union[th.device, str] = "cpu",
#         gae_lambda: float = 1,
#         gamma: float = 0.99,
#         n_envs: int = 1,
#     ):

#         super().__init__(buffer_size, observation_dim, action_dim, device, n_envs=n_envs)
#         self.gae_lambda = gae_lambda
#         self.gamma = gamma
#         self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
#         self.returns, self.episode_starts, self.values, self.log_probs = None, None, None, None
#         self.generator_ready = False
#         self.reset()

#     def reset(self) -> None:

#         self.observations = np.zeros((self.buffer_size, self.n_envs, self.obervaio), dtype=np.float32)
#         self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
#         self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
#         self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
#         self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
#         self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
#         self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
#         self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
#         self.generator_ready = False
#         super().reset()

#     def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
#         """
#         Post-processing step: compute the lambda-return (TD(lambda) estimate)
#         and GAE(lambda) advantage.

#         Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
#         to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
#         where R is the sum of discounted reward with value bootstrap
#         (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

#         The TD(lambda) estimator has also two special cases:
#         - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
#         - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

#         For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

#         :param last_values: state value estimation for the last step (one for each env)
#         :param dones: if the last step was a terminal step (one bool for each env).
#         """
#         # Convert to numpy
#         last_values = last_values.clone().cpu().numpy().flatten()

#         last_gae_lam = 0
#         for step in reversed(range(self.buffer_size)):
#             if step == self.buffer_size - 1:
#                 next_non_terminal = 1.0 - dones
#                 next_values = last_values
#             else:
#                 next_non_terminal = 1.0 - self.episode_starts[step + 1]
#                 next_values = self.values[step + 1]
#             delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
#             last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
#             self.advantages[step] = last_gae_lam
#         # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
#         # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
#         self.returns = self.advantages + self.values

#     def add(
#         self,
#         obs: np.ndarray,
#         action: np.ndarray,
#         reward: np.ndarray,
#         episode_start: np.ndarray,
#         value: th.Tensor,
#         log_prob: th.Tensor,
#     ) -> None:
#         """
#         :param obs: Observation
#         :param action: Action
#         :param reward:
#         :param episode_start: Start of episode signal.
#         :param value: estimated value of the current state
#             following the current policy.
#         :param log_prob: log probability of the action
#             following the current policy.
#         """
#         if len(log_prob.shape) == 0:
#             # Reshape 0-d tensor to avoid error
#             log_prob = log_prob.reshape(-1, 1)

#         self.observations[self.pos] = np.array(obs).copy()
#         self.actions[self.pos] = np.array(action).copy()
#         self.rewards[self.pos] = np.array(reward).copy()
#         self.episode_starts[self.pos] = np.array(episode_start).copy()
#         self.values[self.pos] = value.clone().cpu().numpy().flatten()
#         self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
#         self.pos += 1
#         if self.pos == self.buffer_size:
#             self.full = True

#     def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
#         assert self.full, ""
#         indices = np.random.permutation(self.buffer_size * self.n_envs)
#         # Prepare the data
#         if not self.generator_ready:

#             _tensor_names = [
#                 "observations",
#                 "actions",
#                 "values",
#                 "log_probs",
#                 "advantages",
#                 "returns",
#             ]

#             for tensor in _tensor_names:
#                 self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
#             self.generator_ready = True

#         # Return everything, don't create minibatches
#         if batch_size is None:
#             batch_size = self.buffer_size * self.n_envs

#         start_idx = 0
#         while start_idx < self.buffer_size * self.n_envs:
#             yield self._get_samples(indices[start_idx : start_idx + batch_size])
#             start_idx += batch_size

#     def _get_samples(self, batch_inds: np.ndarray) -> RolloutBufferSamples:
#         data = (
#             self.observations[batch_inds],
#             self.actions[batch_inds],
#             self.values[batch_inds].flatten(),
#             self.log_probs[batch_inds].flatten(),
#             self.advantages[batch_inds].flatten(),
#             self.returns[batch_inds].flatten(),
#         )
#         return RolloutBufferSamples(*tuple(map(self.to_torch, data)))