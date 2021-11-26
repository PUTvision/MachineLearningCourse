from typing import Tuple, List
from collections import namedtuple, deque
import time
import random
import os

import gym
import numpy as np

import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
from torch.utils.data import IterableDataset, DataLoader


class DQN(nn.Module):
    def __init__(self, observation_size: int, number_of_actions: int, hidden_size: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, number_of_actions)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x.float())


Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.bool), np.array(next_states))


class RLDataset(IterableDataset):
    def __init__(self, buffer: ReplayBuffer, sample_size: int):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Tuple:
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]


class Agent:
    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer):
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()

    def reset(self) -> None:
        """Resets the environment and updates the state"""
        self.state = self.env.reset()

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        """
        Using the given network, decide what action to carry out
        using an epsilon-greedy policy
        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device
        Returns:
            action
        """
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = torch.tensor([self.state])
            if device not in ['cpu']:
                state = state.cuda(device)
            q_values = net(state)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())
        return action

    @torch.no_grad()
    def play_step(self, net: nn.Module, epsilon: float = 0.0, device: str = 'cpu') -> Tuple[float, bool]:
        """
        Carries out a single interaction step between the agent and the environment
        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device
        Returns:
            reward, done
        """
        action = self.get_action(net, epsilon, device)
        # do step in the environment
        new_state, reward, done, _ = self.env.step(action)
        exp = Experience(self.state, action, reward, done, new_state)
        self.replay_buffer.append(exp)
        self.state = new_state
        if done:
            self.reset()
        return reward, done


class DQNLightning(pl.LightningModule):
    """ Basic DQN Model
    >>> DQNLightning(env="CartPole-v0")
    DQNLightning(
      (net): DQN(
        (net): Sequential(...)
      )
      (target_net): DQN(
        (net): Sequential(...)
      )
    )
    """

    def __init__(
            self,
            env: str,
            replay_size: int = 200,
            warm_start_steps: int = 200,
            gamma: float = 0.99,
            eps_start: float = 1.0,
            eps_end: float = 0.01,
            eps_last_frame: int = 200,
            sync_rate: int = 10,
            lr: float = 1e-2,
            episode_length: int = 50,
            batch_size: int = 4,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.replay_size = replay_size
        self.warm_start_steps = warm_start_steps
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_last_frame = eps_last_frame
        self.sync_rate = sync_rate
        self.lr = lr
        self.episode_length = episode_length
        self.batch_size = batch_size
        self.env = gym.make(env)
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n
        self.net = DQN(obs_size, n_actions)
        self.target_net = DQN(obs_size, n_actions)
        self.buffer = ReplayBuffer(self.replay_size)
        self.agent = Agent(self.env, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self.populate(self.warm_start_steps)

    def populate(self, steps: int = 1000) -> None:
        """
        Carries out several random steps through the environment to initially fill
        up the replay buffer with experiences
        Args:
            steps: number of random steps to populate the buffer with
        """
        for i in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes in a state `x` through the network and gets the `q_values` of each action as an output
        Args:
            x: environment state
        Returns:
            q values
        """
        output = self.net(x)
        return output

    def dqn_mse_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Calculates the mse loss using a mini batch from the replay buffer
        Args:
            batch: current mini batch of replay data
        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch
        state_action_values = self.net(states).gather(1, torch.tensor(actions, dtype=torch.int64).unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()
        expected_state_action_values = next_state_values * self.gamma + rewards
        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], nb_batch) -> torch.Tensor:
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch received
        Args:
            batch: current mini batch of replay data
            nb_batch: batch number
        Returns:
            Training loss and log metrics
        """
        device = self.get_device(batch)
        epsilon = max(self.eps_end, self.eps_start - self.global_step + 1 / self.eps_last_frame)
        # step through environment with agent
        reward, done = self.agent.play_step(self.net, epsilon, device)
        self.episode_reward += reward
        # calculates training loss
        loss = self.dqn_mse_loss(batch)
        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0
        # Soft update of target network
        if self.global_step % self.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        self.log('total_reward', torch.tensor(self.total_reward).to(device), prog_bar=True)
        self.log('reward', torch.tensor(reward).to(device), prog_bar=True)
        self.log('steps', torch.tensor(self.global_step).to(device))
        return loss

    def configure_optimizers(self) -> List[optim.Optimizer]:
        """Initialize Adam optimizer"""
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = RLDataset(self.buffer, self.episode_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            sampler=None,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch"""
        return batch[0].device.index if self.on_gpu else 'cpu'


def train(model: pl.LightningModule, epochs: int = 100):
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else None,
        accelerator='dp' if torch.cuda.is_available() else None,
        max_epochs=epochs,
        weights_summary=None
    )
    trainer.fit(model)


def main():
    env_name = 'CartPole-v1'
    model = DQNLightning(
        env=env_name, replay_size=2000, warm_start_steps=2000, episode_length=500, eps_last_frame=2000,
        batch_size=32
    )

    env = gym.make(env_name)  # utworzenie środowiska
    for _ in range(1000):  # kolejne kroki symulacji
        train(model, 300)

        observation = env.reset()  # reset środowiska do stanu początkowego
        env.render()  # renderowanie obrazu
        done = False
        steps = 0
        while not done:
            response = model(torch.from_numpy(observation))
            action = torch.argmax(response).item()
            observation, reward, done, info = env.step(action)  # wykonanie akcji
            # print(f'observation: {observation}, reward: {reward}, info: {info}')
            # os.system('cls')
            env.render()  # renderowanie obrazu
            time.sleep(0.01)
            steps += 1
        if steps >= 500:
            print(f'Win!, steps: {steps}')
        else:
            print(f'Fail!, steps: {steps}')
    env.close()  # zamknięcie środowiska


if __name__ == '__main__':
    main()
