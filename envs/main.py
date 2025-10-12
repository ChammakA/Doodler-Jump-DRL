import pygame
import random

WIDTH, HEIGHT = 800, 600
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
            if (self.x + self.width/4 >= platform.x) and (self.x + self.width/4 <= platform.x + platform.width):
                if (self.y + self.height >= platform.y) and (self.y + self.height <= platform.y + platform.height):
                    return True
        return False
    
    def jump(self):
        self.dy = -23
        jump_sound.play()
    
    def move(self):
        self.dy += 0.8
        self.y += self.dy
        self.x += self.dx

        if self.x > WIDTH:
            self.x = 0
        elif self.x < -self.width:
            self.x = WIDTH
        
class Platform:
    def __init__(self, x, y, width=100, height=20, colour=GREEN):
        self.x = x
        self.y = y
        self.width = width
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
platforms = [
    Platform(WIDTH//2, HEIGHT - 30),
    MovingPlatform(WIDTH//3*2, HEIGHT//5*1),
    Platform(WIDTH//4*1, HEIGHT//5*2),
    BreakablePlatform(WIDTH//3*1, HEIGHT//5*3),
    Platform(WIDTH//4*2, HEIGHT//5*4)
]

running = True
while running:
    clock.tick(FPS)
    screen.fill((10, 10, 30))
    for i in range(30):
        pygame.draw.circle(screen, (200, 200, 255), (random.randint(0, WIDTH), random.randint(0, HEIGHT)), 2)


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
            break
    
    # gravity = 1 + (doodler.score / 100)
    # doodler.dy += gravity / 2


    doodler.move()

    if doodler.y < HEIGHT // 2 and doodler.dy < 0:
        scroll = min(-doodler.dy, 15)
        for platform in platforms:
            platform.y += scroll
            if platform.y > HEIGHT:
                platform.y = random.randint(0, WIDTH - platform.width)
                platform.x = random.randint(-50, 0)
                doodler.score += 1
    
    
    

    # draw everything
    doodler.show()
    for platform in platforms:
        platform.show()

    score_text = font.render(str(doodler.score), True, WHITE)
    screen.blit(score_text, (10, 10))

    # game over condition
    if doodler.y > HEIGHT:
        game_over_text = font.render("Game Over!", True, WHITE)
        screen.blit(game_over_text, (50, HEIGHT // 2))
        pygame.display.flip()
        pygame.time.delay(2000)
        running = False
    
    pygame.display.flip()
pygame.quit()