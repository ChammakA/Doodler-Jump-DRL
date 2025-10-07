import pygame
import random

WIDTH, HEIGHT = 800, 600
FPS = 60

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Doodle Jump Clone")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 40)

#Colors
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
    
    def lands(self, platform):
        if (self.dy > 0):
            if (self.x + self.width/4 >= platform.x) and (self.x + self.width/4 <= platform.x + platform.width):
                if (self.y + self.height >= platform.y) and (self.y + self.height <= platform.y + platform.height):
                    return True
        return False
    
    def jump(self):
        self.dy = -20
    
    def move(self):
        self.dy += 1
        self.y += self.dy
        self.x += self.dx

        if self.x > WIDTH:
            self.x = 0
        elif self.x < -self.width:
            self.x = WIDTH
        
class Platform:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 100
        self.height = 20
    
    def show(self):
        pygame.draw.rect(screen, GREEN, (self.x, self.y, self.width, self.height))
    
    # def move(self, dy):
    #     self.y += dy

doodler = Doodler()
platforms = [
    Platform(WIDTH//2, HEIGHT - 30),
    Platform(WIDTH//3*2, HEIGHT//5*1),
    Platform(WIDTH//4*1, HEIGHT//5*2),
    Platform(WIDTH//3*1, HEIGHT//5*3),
    Platform(WIDTH//4*2, HEIGHT//5*4)
]

running = True
while running:
    clock.tick(FPS)
    screen.fill(BLACK)


    # Event Handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        doodler.dx = -5
    elif keys[pygame.K_RIGHT]:
        doodler.dx = 5
    else:
        doodler.dx = 0

    # Game Logic
    for platform in platforms:
        if doodler.lands(platform):
            doodler.jump()
            break
    
    doodler.move()

    if doodler.y < HEIGHT // 2 and doodler.dy < 0:
        for platform in platforms:
            platform.y -= doodler.dy
            if platform.y > HEIGHT:
                platform.x = random.randint(0, WIDTH - 100)
                platform.y = random.randint(-50, 0)
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