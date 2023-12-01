# ~/anaconda3/envs/Torch python3.9.18
# -*- coding: utf-8 -*-

import cv2
from ultralytics import YOLO
from YOLOv8_train import find_FinalVersion

if __name__ == '__main__':
    
    try: # 尋找版本
        version  = find_FinalVersion('train')
        print(f'目前使用第{version}代測試')

    except FileNotFoundError: # 訓練第一代
        print('請先訓練第一代模型')
        exit(0)

    except Exception as e:  # 例外狀況
        print(e)

    else:
        model = YOLO(f'models/G{version}/best.pt')
        method = input('要以哪種方式測試?\n1.圖片或影片\n2.實時鏡頭\n')
        if method == '1':  # 以圖片或影片測試
            target = input('請貼上測試檔案的完整路徑(須為jpg、mp4): ')
            result = model.predict(source=f'{target}', save=True)
        elif method == '2':  # 鏡頭實時測試
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                print('camera not found')
                exit()

            while True:
                success, frame = camera.read()
                if not success:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break

                result = model.predict(source=frame)
                annotated_frame = result[0].plot()
                infoText = "Press q to exit"
                cv2.putText(annotated_frame, infoText, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1)
                cv2.imshow('test', annotated_frame)

                if cv2.waitKey(1) == ord('q'):
                    break

            camera.release()
            cv2.destroyAllWindows()
else:
    print('error')
exit()