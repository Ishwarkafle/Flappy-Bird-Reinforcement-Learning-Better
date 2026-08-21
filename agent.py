import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory 
import itertools
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import random

def get_device() -> torch.device:
    """Determine the optimal available computation device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = get_device()
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)


from config import DQNConfig


class Agent:
    """DQN Agent responsible for training, evaluation, and policy optimization."""

    def __init__(self, param_set: str):
        """Load hyperparameters using DQNConfig dataclass and initialize agent state."""
        self.param_set = param_set
        self.config = DQNConfig.load_from_yaml("parameters.yaml", param_set)

        self.alpha = self.config.alpha
        self.gamma = self.config.gamma

        self.epsilon_init = self.config.epsilon_init
        self.epsilon_min = self.config.epsilon_min
        self.epsilon_decay = self.config.epsilon_decay

        self.replay_memory_size = self.config.replay_memory_size
        self.mini_batch_size = self.config.mini_batch_size

        self.reward_threshold = self.config.reward_threshold
        self.network_sync_rate = self.config.network_sync_rate

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")


    def run(self, is_training: bool = True, render: bool = False, max_episodes: int = None):
        """Execute the agent interaction loop for training or evaluation."""


        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0] # input dim
        num_actions = env.action_space.n # output dim

        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            target_dqn = DQN(num_states, num_actions).to(device)
            # copy the wt & bias vals from policy => target
            target_dqn.load_state_dict(policy_dqn.state_dict())

            steps = 0

            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)

            best_reward = float("-inf")

        else:
            # best policy load
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
            policy_dqn.eval()
            epsilon = 0

        for episode in itertools.count():
            if max_episodes is not None and episode >= max_episodes:
                print(f"Reached max episodes ({max_episodes}). Stopping.")
                break

            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            episode_reward = 0
            terminated = False

            while (not terminated and episode_reward < self.reward_threshold):
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample() # explore
                    action = torch.tensor(action, dtype=torch.long, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax() # exploit

                next_state, reward, terminated, _, _ = env.step(action.item())
                
                episode_reward += reward

                # create tensors
                reward = torch.tensor(reward, dtype=torch.float, device=device)
                next_state = torch.tensor(next_state, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, action, next_state, reward, terminated))
                    steps += 1

                state = next_state
                
            print(f"episode={episode+1} with total reward={episode_reward} & epsilon={epsilon}")

            if is_training:
                # epsilon decay
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                if episode_reward > best_reward:
                    log_msg = f"best reward = {episode_reward} for episode={episode+1}"

                    with open(self.LOG_FILE, "a") as f:
                        f.write(log_msg + "\n")

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward


            if is_training and len(memory) > self.mini_batch_size:
                # get sample
                mini_batch = memory.sample(self.mini_batch_size)

                self.optimize(mini_batch, policy_dqn, target_dqn)

                # sync the network
                if steps > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    steps = 0

            # env.close() - manually stop

    
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # get batch of experiences
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        # calculate target Q-values - if terminations=true => zero
        with torch.no_grad():
            target_q = rewards + (1-terminations) * self.gamma * target_dqn(next_states).max(dim=1)[0]

            
        # calculate y_pred i.e. Q-value from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # compute loss
        loss = self.loss_fn(current_q, target_q)

        # optimize model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), max_norm=1.0)
        self.optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or evaluate a Deep Q-Network (DQN) agent on Flappy Bird."
    )
    parser.add_argument(
        "hyperparameters",
        type=str,
        help="Key name of hyperparameter set in parameters.yaml (e.g., flappybirdv0)",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Enable training mode. If not passed, runs evaluation with rendering.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100000,
        help="Maximum episodes to train or evaluate (default: 100,000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    dql = Agent(param_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True, max_episodes=args.episodes)
    else:
        dql.run(is_training=False, render=True, max_episodes=args.episodes)