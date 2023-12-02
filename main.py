import cv2
import pygame
from random import randint
from ultralytics import YOLO

WIDTH = 640
HEIGHT = 640
FPS = 30
model = YOLO('models/G1/best.pt')

def draw_text(surf, text, size , x, y, color):
    font =pygame.font.SysFont('arial', size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    text_rect.centerx = x
    text_rect.centery = y
    surf.blit(text_surface, text_rect)
    return

def predict(frame):
    cls = None
    result = model.predict(source=frame, max_det=1)
    if len(result[0].boxes.cls) != 0:
        cls = int(result[0].boxes.cls.item())
    frame = result[0].plot()[::-1,:,::-1]

    return (frame, cls)

def frame_transform(frame, scale, angle):
    img = pygame.surfarray.make_surface(frame)
    img = pygame.transform.scale(img, scale)
    img = pygame.transform.rotate(img, angle)
    return img


COUNTDOWN = pygame.USEREVENT + 1

if __name__ == '__main__':
    pygame.init()
    pygame.display.set_caption("黑白猜")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    surface_info = pygame.surface.Surface((WIDTH-400, 300))
    surface_hint = pygame.surface.Surface((WIDTH, HEIGHT-300))
    surface_hint.fill((255,255,255))
    surface_info.fill((25,100,55))


    clock = pygame.time.Clock()
    clock.tick(FPS)

    camera = cv2.VideoCapture(0)

    running = True
    waiting = True
    count = 3
    rnd = 0
    win = 0
    lose = 0
    tie = 0

    while running:

        # waiting start
        while waiting:
            screen.fill((100,100,100))
            draw_text(screen, 'START', 100, WIDTH/2, HEIGHT/2, (255,255,255))
            pygame.display.update()
            first_time = True

            # get input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                elif event.type == pygame.MOUSEBUTTONUP:
                    waiting = False
                    pygame.time.set_timer(COUNTDOWN, 1000)


        # 讀取鏡頭
        success, frame = camera.read()

        if not success:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # 預測
        frame, cls = predict(frame)

        # 事件監測
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == COUNTDOWN:
                count -= 1

        if count == 0:
            pygame.time.set_timer(COUNTDOWN, 0)
            while cls == None:
                success, frame = camera.read()
                if not success:
                    print("Can't receive frame (stream end?). Exiting ...")
                    exit()
                frame, cls = predict(frame)
                img = frame_transform(frame, (300, 400), 90)
                screen.blit(img, (0,HEIGHT-300))
                pygame.display.update()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit()

            computer = randint(0,2) # 0:scissor 1:paper 2:stone

            if computer == cls: # 平手
                tie += 1
            elif (computer == 0 and cls == 1) or (computer == 1 and cls == 2) or (computer == 2 and cls == 0):
                lose += 1
            elif (computer == 0 and cls == 2) or (computer == 1 and cls == 0) or (computer == 2 and cls == 1):
                win += 1
            rnd += 1

            count = 3
            pygame.time.set_timer(COUNTDOWN, 1000)


        surface_info.fill((25,100,55))
        surface_hint.fill((255,255,255))
        draw_text(surface_hint, str(count), 48, WIDTH/2, 170, (0,0,0))
        draw_text(surface_info, f'round: {rnd}', 36, 120, 60, (255,255,255))
        draw_text(surface_info, f'win: {win}', 36, 120, 120, (255,255,255))
        draw_text(surface_info, f'lose: {lose}', 36, 120, 180, (255,255,255))
        draw_text(surface_info, f'tie: {tie}', 36, 120, 240, (255,255,255))
        frame = frame_transform(frame, (300, 400), 90)

        screen.blit(surface_hint, (0,0))
        screen.blit(surface_info, (400, HEIGHT-300))
        screen.blit(frame, (0,HEIGHT-300))
        pygame.display.update()

    exit()