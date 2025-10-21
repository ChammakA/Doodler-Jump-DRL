import pygame
import sys
from envs.doodler_env import DoodlerEnv

def main():
    env = DoodlerEnv(render_mode="human", seed=42)  # human rendering enabled
    obs, info = env.reset()
    clock = pygame.time.Clock()

    running = True
    while running:
        # --- Handle keyboard input ---
        action = 2  # Default: no movement
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                env.close()
                sys.exit()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            print("Quitting game...")
            running = False
            env.close()
            sys.exit()
        elif keys[pygame.K_LEFT]:
            action = 0  # Move left
        elif keys[pygame.K_RIGHT]:
            action = 1  # Move right

        # --- Step environment with chosen action ---
        obs, reward, done, trunc, info = env.step(action)

        # --- Render environment frame ---
        env.render()
        clock.tick(60)  # 60 FPS

        # --- Reset if game over ---
        if done or trunc:
            print(f"Game Over! Final Score: {info['score']}")
            obs, info = env.reset()

if __name__ == "__main__":
    main()
