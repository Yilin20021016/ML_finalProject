from ultralytics import YOLO
import cv2

model = YOLO('models/G1.pt')
result = model.predict(source='./dataset/test/images/IMG_2119.jpg', save=True)


# camera = cv2.VideoCapture(0)
# if not camera.isOpened():
#     print('camera not found')
#     exit()

# while True:
#     success, frame = camera.read()
#     if not success:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break

#     result = model.predict(source=frame)
#     annotated_frame = result[0].plot()
    


#     cv2.imshow('test', annotated_frame)

#     if cv2.waitKey(1) == ord('q'):
#         break

# camera.release()
# cv2.destroyAllWindows()
