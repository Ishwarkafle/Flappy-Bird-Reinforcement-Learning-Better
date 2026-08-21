import argparse
import os
import torch
import gymnasium as gym
import flappy_bird_gymnasium
from dqn import DQN


def evaluate_agent(
    model_path: str = "runs/flappybirdv0.pt",
    num_episodes: int = 10,
    render: bool = True
) -> None:
    """Evaluate a trained DQN agent across multiple episodes and print statistics.
    
    Args:
        model_path (str): Path to the PyTorch model checkpoint (.pt).
        num_episodes (int): Total number of evaluation episodes to play.
        render (bool): Whether to render the game visually in real time.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint file '{model_path}' not found.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("FlappyBird-v0", render_mode="human" if render else None)

    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n

    policy_dqn = DQN(num_states, num_actions).to(device)
    policy_dqn.load_state_dict(torch.load(model_path, map_location=device))
    policy_dqn.eval()

    rewards = []

    print(f"Starting evaluation of '{model_path}' over {num_episodes} episodes...")

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float, device=device)
        episode_reward = 0.0
        done = False

        while not done:
            with torch.no_grad():
                action = policy_dqn(state_tensor.unsqueeze(0)).squeeze().argmax().item()

            next_state, reward, done, _, _ = env.step(action)
            state_tensor = torch.tensor(next_state, dtype=torch.float, device=device)
            episode_reward += reward

        rewards.append(episode_reward)
        print(f"Episode {ep}/{num_episodes} - Total Reward: {episode_reward:.2f}")

    env.close()

    avg_reward = sum(rewards) / len(rewards)
    max_reward = max(rewards)
    min_reward = min(rewards)

    print("\n--- Evaluation Summary ---")
    print(f"Episodes Evaluated: {num_episodes}")
    print(f"Average Reward:     {avg_reward:.2f}")
    print(f"Maximum Reward:     {max_reward:.2f}")
    print(f"Minimum Reward:     {min_reward:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Flappy Bird DQN agent.")
    parser.add_argument("--model", type=str, default="runs/flappybirdv0.pt", help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable visual rendering")
    args = parser.parse_args()

    evaluate_agent(
        model_path=args.model,
        num_episodes=args.episodes,
        render=not args.no_render
    )
