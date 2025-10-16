import pygame
import random

WIDTH, HEIGHT = 500, 600
FPS = 60

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Doodle Jump Clone")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 40)

#Sound Effects
jump_sound = pygame.mixer.Sound("Jump_Sound.mp3")
pygame.mixer.music.load("Background_Music.mp3")
pygame.mixer.music.play(-1)  # Loop indefinitely

#Colours
PINK = (255, 192, 203)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class Doodler:
    def __init__(self):
        self.x = WIDTH // 2
        self.y = HEIGHT - 70
        self.width = 40
        self.height = 40
        self.dy = 0
        self.dx = 0
        self.score = 0
    
    def show(self):
        pygame.draw.ellipse(screen, PINK, (self.x, self.y, self.width, self.height))

        if self.x < 0:
            pygame.draw.ellipse(screen, PINK, (WIDTH + self.x, self.y, self.width, self.height))
        elif (self.x + self.width > WIDTH):
            pygame.draw.ellipse(screen, PINK, (self.x - WIDTH, self.y, self.width, self.height))
    
    def lands(self, platform):
        if (self.dy > 0):
            if (self.x + self.width * 0.2 < platform.x + platform.width) and (self.x + self.width * 0.8 > platform.x):
                if (self.y + self.height >= platform.y) and (self.y + self.height <= platform.y + platform.height):
                    return True
        return False
    
    def jump(self):
        self.dy = -18
        jump_sound.play()
    
    def move(self):
        self.dy += 1.0
        self.y += self.dy
        self.x += self.dx

        if self.x > WIDTH:
            self.x = 0
        elif self.x < -self.width:
            self.x = WIDTH
        
class Platform:
    def __init__(self, x, y, width=None, height=20, colour=GREEN):
        self.x = x
        self.y = y
        if width is None:
            self.width = random.randint(70, 120)
        self.height = height
        self.colour = colour

    
    def show(self):
        pygame.draw.rect(screen, GREEN, (self.x, self.y, self.width, self.height))
    
    def update(self, dy):
        self.y += dy
    
class MovingPlatform(Platform):
    def __init__(self, x, y):
        super().__init__(x, y, colour=(0, 200, 255))
        self.speed = random.choice([-2, 2])

    def update(self, dy):
        super().update(dy)
        self.x += self.speed
        if self.x <= 0 or self.x + self.width >= WIDTH:
            self.speed *= -1

class BreakablePlatform(Platform):
    def __init__(self, x, y):
        super().__init__(x, y, colour=(255, 0, 0))
        self.broken = False

    def show(self):
        if not self.broken:
            super().show()
    
    def break_platform(self):
        self.broken = True

doodler = Doodler()
platforms = []

starting_platform = Platform(WIDTH // 2 - 50, HEIGHT - 30)
platforms.append(starting_platform)

start_y = HEIGHT - 100
while start_y > -HEIGHT:
    x = random.randint(0, WIDTH - 100)
    if (abs(x - starting_platform.x) > 80) or (abs(start_y - starting_platform.y) > 50):
        platforms.append(Platform(x, start_y))
    start_y -= random.randint(80, 120)

max_platform = HEIGHT
highest_platform = HEIGHT

running = True
while running:
    clock.tick(FPS)
    for i in range(HEIGHT):
        color = (10, 10, 30 + i // 4)
        pygame.draw.line(screen, color, (0, i), (WIDTH, i))
    for i in range(15):
        pygame.draw.circle(screen, (180, 180, 255), (random.randint(0, WIDTH), random.randint(0, HEIGHT)), 1)


    # Event Handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    MOVE_SPEED = 8 # hortizontal move speed

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        doodler.dx = -MOVE_SPEED
    elif keys[pygame.K_RIGHT]:
        doodler.dx = MOVE_SPEED
    else:
        doodler.dx *= 0.9  # friction effect

    # Game Logic
    for platform in platforms:
        if doodler.lands(platform):
            doodler.jump()
            
            if (platform.y < highest_platform):
                doodler.score += 1
                highest_platform = platform.y
            break
    
    doodler.move()

    if doodler.y < HEIGHT // 2 and doodler.dy < 0:
        scroll = min(-doodler.dy, 10)
        
        for platform in platforms:
            platform.y += scroll

        max_platform = min(platform.y for platform in platforms)
        vertical_gap = 80
        if max_platform > 0:
            new_y = max_platform - vertical_gap
            attempts = 0
            while True:
                new_x = random.randint(0, WIDTH - 100)
                if all(abs(new_x - platform.x) > 80 for platform in platforms if abs(platform.y - new_y) < 10):
                    break
                attempts += 1
                if attempts > 10:  # prevent infinite loop
                    break
                

            platform_type = random.choices(
                [Platform, MovingPlatform, BreakablePlatform],
                weights=[0.6, 0.25, 0.15],
            )[0]

            if platform_type == Platform:
                new_platform = Platform(new_x, new_y)
            elif platform_type == MovingPlatform:
                new_platform = MovingPlatform(new_x, new_y)
            else:
                new_platform = BreakablePlatform(new_x, new_y)

            platforms.append(new_platform)
            
        platforms = [p for p in platforms if p.y < HEIGHT + 40]
    
    if doodler.score % 10 == 0 and doodler.score != 0:
        MOVE_SPEED = 8 + doodler.score // 20
        GRAVITY = 0.8 + doodler.score / 200
    
    
    

    # draw everything
    doodler.show()
    for platform in platforms:
        platform.show()

    score_text = font.render(str(doodler.score), True, WHITE)
    screen.blit(score_text, (10, 10))

    # game over condition
    if doodler.y + doodler.height > HEIGHT:
        game_over_text = font.render("Game Over!", True, WHITE)
        screen.blit(game_over_text, (50, HEIGHT // 2))
        pygame.display.flip()
        pygame.time.delay(2000)
        running = False
    
    pygame.display.flip()
pygame.quit()