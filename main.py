import cv2
import pygame
from random import randint
from ultralytics import YOLO

# const
WIDTH = 640
HEIGHT = 640
FPS = 30
WHITE = (255,255,255)
COUNTDOWN = pygame.USEREVENT + 1

# variable
model_part1 = YOLO('models/G1/best.pt')
model_part2 = YOLO('models/G1/best.pt') # temp
chart_part1 = {0:'siccor', 1:'paper', 2:'stone'}
chart_part2 = {0:'up', 1:'down', 2:'right', 3:'left'} # temp

# funciton
def draw_text(surf, text, size , x, y, color):
    font =pygame.font.SysFont('arial', size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    text_rect.centerx = x
    text_rect.centery = y
    surf.blit(text_surface, text_rect)
    return

def predict(model,frame):
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

def surface_refresh(string, rnd, win, lose, frame):
    # clean all surface
    surface_hint.fill(WHITE)
    surface_info.fill((25,100,55))
    # show text
    draw_text(surface_hint, str(string), 48, WIDTH/2, 170, (0,0,0))
    draw_text(surface_info, f'round: {rnd}', 36, 120, 60, WHITE)
    draw_text(surface_info, f'win: {win}', 36, 120, 120, WHITE)
    draw_text(surface_info, f'lose: {lose}', 36, 120, 180, WHITE)
    frame = frame_transform(frame, (300, 400), 90)
    # screen blit
    screen.blit(surface_hint, (0,0))
    screen.blit(surface_info, (400, HEIGHT-300))
    screen.blit(frame, (0,HEIGHT-300))
    pygame.display.update()
    return


if __name__ == '__main__':
    # initialize setting
    pygame.init()
    pygame.display.set_caption("黑白猜")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    clock.tick(FPS)
    camera = cv2.VideoCapture(0)

    # initialize variable
    running = True
    waiting = True
    count = 5
    rnd = 1
    win = 0
    lose = 0
    stage = 1
    status = ''

    # create surface
    surface_info = pygame.surface.Surface((WIDTH-400, 300))
    surface_hint = pygame.surface.Surface((WIDTH, HEIGHT-300))

    # answer
    computer = (randint(0,2), randint(0,3))

    while running:
        # waiting start
        while waiting:
            screen.fill((100,100,100))
            draw_text(screen, 'START', 100, WIDTH/2, HEIGHT/2, WHITE)
            # event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                elif event.type == pygame.MOUSEBUTTONUP:
                    waiting = False
                    pygame.time.set_timer(COUNTDOWN, 1000)
            pygame.display.update()

        # read camera
        success, frame = camera.read()   
        if not success:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        # predict
        if stage == 1:
            frame, cls = predict(model_part1, frame)
        elif stage == 2:
            frame, cls = predict(model_part2, frame)
        
        # event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == COUNTDOWN:
                count -= 1

        # part1
        if count == 0 and stage == 1:
            pygame.time.set_timer(COUNTDOWN, 0)
            # answer checking
            while cls == None:
                success, frame = camera.read()
                if not success:
                    print("Can't receive frame (stream end?). Exiting ...")
                    exit()
                frame, cls = predict(model_part1, frame)
                surface_refresh(count, rnd, win, lose, frame)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit()

            # show answer
            start_time = pygame.time.get_ticks()
            end_time = pygame.time.get_ticks()
            while (end_time-start_time)/1000 < 1:
                end_time=pygame.time.get_ticks()
                surface_refresh(chart_part1[computer[0]], rnd, win, lose, frame)

            # win or lose checking
            if (computer[0] == 0 and cls == 2) or (computer[0] == 1 and cls == 0) or (computer[0] == 2 and cls == 1): # win
                stage = 2
                status = 'win'
            elif (computer[0] == 0 and cls == 1) or (computer[0] == 1 and cls == 2) or (computer[0] == 2 and cls == 0): # lose
                stage = 2
                status = 'lose'
            elif computer[0] == cls: # tie
                rnd += 1
                computer = (randint(0,2), randint(0,3))

            count = 5
            pygame.time.set_timer(COUNTDOWN, 1000)
            continue

        # part2
        if count == 0 and stage == 2:
            pygame.time.set_timer(COUNTDOWN, 0)
            # answer checking
            while cls == None:
                success, frame = camera.read()
                if not success:
                    print("Can't receive frame (stream end?). Exiting ...")
                    exit()
                frame, cls = predict(model_part2, frame)
                surface_refresh(count, rnd, win, lose, frame)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit()

            # show answer
            start_time = pygame.time.get_ticks()
            end_time = pygame.time.get_ticks()
            while (end_time-start_time)/1000 < 1:
                end_time=pygame.time.get_ticks()
                surface_refresh(chart_part1[computer[0]], rnd, win, lose, frame)

            # win or lose checking
            if computer[1] == cls:
                if status == 'win':
                    win += 1
                elif status == 'lose':
                    lose += 1
                surface_refresh('You '+status, rnd, win, lose, frame)
            rnd += 1
            computer = (randint(0,2), randint(0,3))
            stage = 1
            count = 5
            pygame.time.set_timer(COUNTDOWN, 1000)
            continue

        surface_refresh(count, rnd, win, lose, frame)
    exit()