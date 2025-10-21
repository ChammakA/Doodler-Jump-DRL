"""
doodler_env.py
----------------
A custom Gymnasium-compatible reinforcement learning environment 
based on the classic "Doodle Jump" game mechanics.

This environment is designed for training RL agents using algorithms 
like PPO or DQN. It supports:
- Continuous procedural platform generation
- Collision detection and landing logic
- Reward shaping for height progression, landings, and alignment
- Pygame-based rendering for visualization

Classes:
    Doodler: The player character with jump physics and wrap-around behavior.
    Platform: A stationary platform the doodler can land on.
    MovingPlatform: A horizontally moving variant of Platform.
    DoodlerEnv: The main environment class implementing the Gymnasium API.
"""
import random
import numpy as np
import pygame
from gymnasium import Env, spaces

# Game Constants
WIDTH, HEIGHT = 500, 600
FPS = 60
GRAVITY = 0.7
JUMP_STRENGTH = -18
MOVE_SPEED = 5
SCROLL_TRIGGER_RATIO = 0.35
FRAME_LIMIT = 2000

PINK = (255, 192, 203)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

#-----------------------
#     DOODLER CLASS
#-----------------------

class Doodler:
    '''Represents the main player character (the doodler)'''
    def __init__(self):
        #Initialize doodler position, size, velocity, etc
        self.x = WIDTH // 2
        self.y = HEIGHT - 70
        self.width = 40
        self.height = 40
        self.dy = 0
        self.dx = 0
        self.score = 0

    def show(self, screen):
        '''Draws the doodler on the screen, with a screen-wrap effect'''
        pygame.draw.ellipse(screen, PINK, (self.x, self.y, self.width, self.height))

        # Wrap around horizontally (appear on opposite side)
        if self.x < 0:
            pygame.draw.ellipse(screen, PINK, (WIDTH + self.x, self.y, self.width, self.height))
        elif (self.x + self.width > WIDTH):
            pygame.draw.ellipse(screen, PINK, (self.x - WIDTH, self.y, self.width, self.height))
    
    def lands(self, platform):
        '''Checks if the doodler has landed on a given platform'''
        if (self.dy > 0):
            if (self.x + self.width * 0.2 < platform.x + platform.width) and (self.x + self.width * 0.8 > platform.x):
                if (self.y + self.height >= platform.y) and (self.y + self.height <= platform.y + platform.height):
                    return True
        return False
        
    def jump(self):
        '''Make the doodler jump by setting a vertical velocity'''
        self.dy = JUMP_STRENGTH
        
    def move(self):
        '''Updates doodler position based on velocity and gravity'''
        self.dy += GRAVITY
        self.y += self.dy
        self.x += self.dx

        # Wrap horizontally across screen
        if self.x > WIDTH:
            self.x = 0
        elif self.x < -self.width:
            self.x = WIDTH

#-----------------------
#     PLATFORM CLASS
#-----------------------

class Platform:
    '''Represents a stationary platform the doodler can jump on'''

    def __init__(self, x, y, row=None, width=None, height=20, colour=GREEN):
        self.x = x
        self.y = y

        if width is None:
            self.width = random.randint(70, 120)
        else:
            self.width = width

        self.height = height
        self.colour = colour
        self.row = row

    
    def show(self, screen):
        '''Draws the platform on the screen'''
        pygame.draw.rect(screen, self.colour, (self.x, self.y, self.width, self.height))

    def update(self, dy):
        '''Moves the platforms vertically (used for scrolling)'''
        self.y += dy
    
class MovingPlatform(Platform):
    '''A special platform that moves horizontally'''

    def __init__(self, x, y, row=None):
        super().__init__(x, y, row=row, colour=(0, 0, 255))
        self.speed = random.choice([-2, 2])

    def update(self, dy):
        '''Moves borth vertically (scrolling) and horizontally'''
        super().update(dy)
        self.x += self.speed
        
        if self.x <= 0:
            self.x = 0
            self.speed *= -1
        elif self.x + self.width >= WIDTH:
            self.x = WIDTH - self.width
            self.speed *= -1

#-----------------------
#     ENVIRONMENT
#-----------------------

class DoodlerEnv(Env):
    '''Gymnasium environment for Doodler Jump, suitable for DRL Training'''

    metadata = {'render.modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, render_mode=None, obs_platform=5, frame_skip=1, seed=None, reward_config=None):
        super().__init__()

        # Core environment paramters
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.render_mode = render_mode
        self.obs_platform = obs_platform
        self.frame_skip = frame_skip

        # Action space: 0 = Left, 1 = Right, 2 = No Movement
        self.action_space = spaces.Discrete(3)  

        # Observation space (normalized game state vector)
        obs_len = 2 + 3 * 5
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)

        # Rendering elements
        self.screen = None
        self.clock = None
        self.font = None

        # Reward configuration (customizable)
        self.reward_config = reward_config or {
            "type": "survivor",
            "weights": {
                "height_progress": 5.0,
                "death": -10.0,
                "landing": 1.0,
            }
        }

        if self.render_mode == 'human':
            self._init_render()

        self.seed(seed)
        self.reset()

    # ------- Initialization ---------
    def _init_render(self):
        '''Initializes pygame window and fonts'''

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Doodle Jump - DRL Testing")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 24)
    
    def seed(self, seed=None):
        '''Sets random seeds for reproduceability'''

        self._seed = seed
        random.seed(self._seed)
        np.random.seed(self._seed)
        return [seed]
    
    def _init_game_state(self):
        '''Initializes or resets all game entities and environment variables'''

        # if self._seed is not None:
        #     random.seed(self._seed)
        #     np.random.seed(self._seed)

        # TODO: Temporary set seed until training is configured properly. Remove once completed.
        temp_seed = 42
        random.seed(temp_seed)
        np.random.seed(temp_seed)

        # Doodler and game state
        self.doodler = Doodler()
        self.platforms = []
        self.highest_landed_row = 0
        self.row_counter = 0
        self.episode_steps = 0
        self.episode_score = 0
        self.terminated = False
        self.prev_score = 0
        self.max_height = HEIGHT - 70
        self.steps_landing = 0
        self.last_landed_platform = None
        self.same_platform_landings = 0

        # Base Platform
        start_platform = Platform(self.WIDTH // 2 - 50, self.HEIGHT - 30, row=0)
        self.platforms.append(start_platform)

        start_y = self.HEIGHT - 100
        vertical_gap_min = 80
        vertical_gap_max = 120

        while start_y > -self.HEIGHT:
            self.row_counter += 1
            x = random.randint(0, self.WIDTH - 100)

            plat_type = random.choices(
                [Platform, MovingPlatform],
                weights=[0.7, 0.2],
            )[0]

            if plat_type == Platform:
                new_platform = Platform(x, start_y, row=self.row_counter)   
            else: 
                new_platform = MovingPlatform(x, start_y, row=self.row_counter)
            self.platforms.append(new_platform)
            start_y -= random.randint(vertical_gap_min, vertical_gap_max)
    
    # ---- RESET ------
    def reset(self, seed=None, options=None):
        '''Resets the environment for a new episode.'''

        if seed is not None:
            self.seed(seed)

        self._init_game_state()
        self.prev_dist_x = 0.0
        self.doodler.dx = 0

        observation = self._get_observation()
        info = {'score': self.episode_score, 'steps': self.episode_steps}

        return observation, info
    
    # ---- Action and Physics ----
    def _apply_action(self, action):
        '''Applies player action (left, right, or no move).'''

        if action == 0:  # Move Left
            self.doodler.dx = -MOVE_SPEED
        elif action == 1:  # Move Right
            self.doodler.dx = MOVE_SPEED
        else:  # No Movement
            self.doodler.dx = 0

    def _physics_step(self):
        '''Simulates one frame of game physics, movement, and collisions.'''

        prev_y = self.doodler.y
        self._apply_movement_and_gravity()
        self.steps_landing += 1
        
        landed_plat = self._check_platform_collision(prev_y)
        if landed_plat:
            self._handle_landing(landed_plat, prev_y)

        self._scroll_world()
        self._cleanup_platform()

    def _apply_movement_and_gravity(self):
        self.doodler.move()
        for platform in self.platforms:
            platform.update(0)
    
    def _check_platform_collision(self, prev_y):
        if self.doodler.dy <= 0:
            return None

        for platform in self.platforms:
            horizontal_overlap = (
                self.doodler.x + self.doodler.width * 0.2 < platform.x + platform.width
                and self.doodler.x + self.doodler.width * 0.8 > platform.x
            )
            if not horizontal_overlap:
                continue

            prev_bottom = prev_y + self.doodler.height
            curr_bottom = self.doodler.y + self.doodler.height

            if prev_bottom <= platform.y + 5 and curr_bottom >= platform.y:
                self.doodler.y = platform.y - self.doodler.height
                return platform
        return None

    def _handle_landing(self, landed_plat, prev_y):
        self.steps_landing = 0
        self.doodler.jump()

        if self.last_landed_platform == landed_plat:
            self.same_platform_landings += 1
        else:
            self.same_platform_landings = 0

        if getattr(landed_plat, "row", None) is not None and landed_plat.row > self.highest_landed_row:
            gained = landed_plat.row - self.highest_landed_row
            self.episode_score += gained
            self.highest_landed_row = landed_plat.row

        self.last_landed_platform = landed_plat
        if self.doodler.y < self.max_height:
            self.max_height = self.doodler.y

    def _scroll_world(self):
        '''Scrolls the game world upward when the doodler reaches a threshold height.'''

        if self.doodler.y >= self.HEIGHT * 0.35 or self.doodler.dy >= 0:
            return

        scroll = int(abs(self.doodler.dy)) or 1
        self.doodler.y += scroll
        for platform in self.platforms:
            platform.y += scroll

        min_y = min(p.y for p in self.platforms)
        if min_y > 0:
            self._spawn_new_platform(min_y)

    def _spawn_new_platform(self, min_y):
        new_y = min_y - random.randint(80, 120)
        new_x = random.randint(0, self.WIDTH - 100)
        plat_class = random.choices([Platform, MovingPlatform], weights=[0.7, 0.2])[0]
        self.row_counter += 1
        self.platforms.append(plat_class(new_x, new_y, row=self.row_counter))

    def _cleanup_platform(self):
        self.platforms = [p for p in self.platforms if p.y < self.HEIGHT]

    # ---- Observation ----
    def _get_observation(self):
        '''Generates normalized observation vector for the agent.
            Observation vector includes:
            - Doodler's normalized horizontal (dx) and vertical (dy) velocities
            - Up to 3 upcoming platforms, each described by:
                - Relative horizontal offset to doodler (-1 to 1)
                - Relative vertical distance (0 to 1)
                - Platform type (0 = static, 1 = moving)
                - Platform width normalized by screen width
                - Platform horizontal speed (if moving)
            Missing platforms are padded with zeros.
        '''

        dx = np.clip(self.doodler.dx / 20.0, -1.0, 1.0)
        dy = np.clip(self.doodler.dy / 30.0, -1.0, 1.0)
        obs = [dx, dy]

        # Sort platforms above the doodler (closest first)
        platforms_above = sorted([platform for platform in self.platforms if platform.y < self.doodler.y],
                                key=lambda p: p.y, reverse=True)


        # Encode up to 3 upcoming platforms with normalized data
        for i in range(3):
            if i < len(platforms_above):
                p = platforms_above[i]
                rel_x = ((p.x + p.width/2) - (self.doodler.x + self.doodler.width/2)) / p.width
                rel_x = np.clip(rel_x, -1.0, 1.0)
                rel_y = (self.doodler.y - p.y) / self.HEIGHT
                plat_type = 1.0 if isinstance(p, MovingPlatform) else 0.0
                plat_speed = p.speed / 5.0 if isinstance(p, MovingPlatform) else 0.0
                plat_w = p.width / self.WIDTH
                obs += [np.clip(rel_x, -1.0, 1.0), np.clip(rel_y, 0.0, 1.0), plat_type, plat_w, plat_speed]
            else:
                # Fill missing platforms with zero padding
                obs += [0.0, 1.0, 0.0, 0.0, 0.0]  

        return np.array(obs, dtype=np.float32)
    
    # ---- Step ----
    def step(self, action):
        '''Performs one environment step: apply action, advance physics, and compute reward.'''

        if self.terminated:
            observation = self._get_observation()
            return observation, 0.0, True, False, {'score': self.episode_score, 'steps': self.episode_steps}

        prev_height = self.max_height
        prev_score = self.episode_score

        self._apply_action(action)

        for _ in range(self.frame_skip):
            self._physics_step()
            self.episode_steps += 1

        observation = self._get_observation()

        # Reward system
        reward = self._compute_reward(prev_height, prev_score)
        weights = self.reward_config.get("weights", {})

        # --- Termination conditions ---
        # - If doodler falls below the screen -> death
        # - If frame limit is reached -> force episode end
        if self.doodler.y > self.HEIGHT:
            reward += weights.get("death", -10.0)
            self.terminated = True
            terminated = True
        elif self.episode_steps >= FRAME_LIMIT:
            self.terminated = True
            terminated = True
        else:
            terminated = False

        info = {
            'score': self.episode_score, 
            'steps': self.episode_steps, 
            'highest_landed_row': self.highest_landed_row,
            'max_height': self.max_height
        }

        return observation, float(reward), terminated, False, info

    # - Reward upward motion and height progression
    # - Penalize repeated landings on the same platform
    # - Encourage horizontal alignment with the next platform
    # - Small time penalty to encourage efficient climbing

    # TODO: Improve reward system for better RL agent training
    def _compute_reward(self, prev_height, prev_score):
        reward = 0.0
        weights = self.reward_config.get("weights", {})

        # Upward velocity reward
        reward += max(0, -self.doodler.dy) * 0.05

        # Progress reward
        if self.max_height < prev_height:
            reward += 5.0

        # Penalty for bouncing on the same platform
        if self.same_platform_landings > 1:
            reward -= (0.01 * self.same_platform_landings)

        # Height gain reward
        height_gain = (prev_height - self.max_height) / self.HEIGHT
        if height_gain > 0:
            reward += height_gain * weights.get("height_progress", 100.0)

        # Landing reward
        score_gain = self.episode_score - prev_score
        if score_gain > 0:
            reward += score_gain * weights.get("landing", 20.0)

        # Small time penalty
        reward -= 0.002

        # Horizontal alignment reward
        next_plats = sorted([p for p in self.platforms if p.y < self.doodler.y], key=lambda p: p.y, reverse=True)
        if next_plats:
            next_plat = next_plats[0]
            x_center_doodler = self.doodler.x + self.doodler.width / 2
            x_center_plat = next_plat.x + next_plat.width / 2
            dist_x = abs(x_center_doodler - x_center_plat) / self.WIDTH
            vert_dist = (self.doodler.y - next_plat.y) / self.HEIGHT
            reward += ((1.0 - dist_x) * max(0, 1 - vert_dist)) * 5.0

        return reward
    
    # ---- Rendering ----
    def render(self):
        '''Renders the game visually using Pygame'''

        if self.render_mode is None:
            return

        self._init_render()
        if not self._handle_events:
            return
            
        # Background and stars
        self.screen.fill((10, 10, 30))
        for i in range(15):
            pygame.draw.circle(self.screen, (180, 180, 255), (random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT)), 1)

        # Draw platforms and doodler
        for platform in self.platforms:
            platform.show(self.screen)
        self.doodler.show(self.screen)

        # HUD (score + step count)
        score_text = self.font.render(f"Score: {self.episode_score}", True, WHITE)
        steps_text = self.font.render(f"Steps: {self.episode_steps}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(steps_text, (10, 40))

        pygame.display.flip()
        
        if self.clock:
            self.clock.tick(self.metadata['render_fps'])

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False
        return True
    
    # ---- Cleanup ----
    def close(self):
        '''Properly closes the Pygame window and releases resources.'''
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None