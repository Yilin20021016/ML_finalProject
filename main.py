import cv2
import pygame
from ultralytics import YOLO

WIDTH = 640
HEIGHT = 800
FPS = 30
model = YOLO('models/G1/best.pt')


pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("黑白猜")

clock = pygame.time.Clock()
clock.tick(FPS)

camera = cv2.VideoCapture(0)

if __name__ == '__main__':
    running = True
    while running:
        success, frame = camera.read()
        if not success:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        result = model.predict(source=frame, max_det=1)
        cls = int(result[0].boxes.cls.item())
        annotated_frame = result[0].plot()
        annotated_frame = annotated_frame[::-1,:,::-1]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        img = pygame.surfarray.make_surface(annotated_frame)
        img = pygame.transform.rotate(img, 90)
        screen.blit(img, (0,0))

        pygame.display.update()