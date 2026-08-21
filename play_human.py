import gymnasium as gym
import flappy_bird_gymnasium
import pygame
import sys


def main():
    """Interactive human playable Flappy Bird game launcher."""
    print("Initializing Flappy Bird manual controller...")
    print("Controls: Press SPACEBAR to flap, ESC to quit.")

    env = gym.make("FlappyBird-v0", render_mode="human")
    state, info = env.reset()
    
    clock = pygame.time.Clock()
    running = True
    total_score = 0.0

    while running:
        action = 0  # 0: do nothing, 1: flap

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action = 1
                elif event.key == pygame.K_ESCAPE:
                    running = False

        state, reward, terminated, truncated, info = env.step(action)
        total_score += reward

        if terminated or truncated:
            print(f"Game Over! Final Score: {total_score:.2f}")
            state, info = env.reset()
            total_score = 0.0

        clock.tick(30)  # Maintain 30 FPS frame rate

    env.close()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
