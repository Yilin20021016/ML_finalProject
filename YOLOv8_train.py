# ~/anaconda3/envs/Torch python3.9.18
# -*- coding: uft-8 -*-

from ultralytics import YOLO
from shutil import copy
from os import path, mkdir, listdir

# 尋找最新版本
def find_FinalVersion(set:str) -> str:
    directory = listdir('./runs/detect')
    setList = list(filter(lambda x: set in x, directory))
    version = max(setList).strip(set)
    return version

if __name__ == '__main__':

    try: # 訓練新一代
        version  = find_FinalVersion('train')
        if version == '':
            version = '1'
        print(f'目前訓練為第{version+1}代')

        model = YOLO(f'./models/G{version}/last.pt')
        results = model.train(data="dataset/data.yaml", epochs=300, batch=3, workers=0)
        results = model.val(data="dataset/data.yaml")

        if not path.exists(f'./models/G{version+1}'):
            mkdir(f'./models/G{version+1}')
        copy(f'./runs/detect/train{version+1}/weights/best.pt',
             f'./models/G{version+1}')
        copy(f'./runs/detect/train{version+1}/weights/last.pt',
             f'./models/G{version+1}')
        
        exit(0)
    except FileExistsError: # 訓練第一代
        model = YOLO('./models/yolov8n.pt')
        results = model.train(data="dataset/data.yaml", epochs=300, batch=3, workers=0)
        results = model.val(data="dataset/data.yaml")

        if not path.exists(f'./models/G1'):
            mkdir(f'./models/G1')
        copy(f'./runs/detect/train/weights/best.pt',
             f'./models/G1')
        copy(f'./runs/detect/train/weights/last.pt',
             f'./models/G1')
        
        exit(0)
    # except:
    #     print('請檢察路徑是否正確')
