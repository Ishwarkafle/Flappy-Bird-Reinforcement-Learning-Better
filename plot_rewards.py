import os
import matplotlib.pyplot as plt

def plot_log(log_path="runs/flappybirdv0.log", save_path="runs/reward_progress.png"):
    if not os.path.exists(log_path):
        print(f"Log file '{log_path}' not found.")
        return

    episodes = []
    rewards = []

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "best reward =" not in line:
                continue
            try:
                # Format: "best reward = X for episode=Y"
                parts = line.split()
                reward_val = float(parts[3])
                ep_val = int(parts[-1].split("=")[-1])
                episodes.append(ep_val)
                rewards.append(reward_val)
            except Exception as e:
                continue

    if not episodes:
        print("No reward logs found.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, marker="o", color="#3b82f6", linewidth=2, label="Best Reward")
    plt.title("Flappy Bird DQN - Training Best Reward Progress", fontsize=14, fontweight="bold")
    plt.xlabel("Episode Number", fontsize=12)
    plt.ylabel("Best Reward", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Reward plot saved to '{save_path}'")
    plt.show()

if __name__ == "__main__":
    plot_log()
