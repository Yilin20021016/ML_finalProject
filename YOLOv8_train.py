# ~/anaconda3/envs/Torch python3.9.18
# -*- coding: utf-8 -*-

import os
from os import path, mkdir, listdir
from shutil import copy
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 尋找最新版本
def find_FinalVersion(set:str) -> str:
    directory = listdir('./runs/detect')
    setList = list(filter(lambda x: set in x, directory))
    version = max(setList).strip(set)
    return version

if __name__ == '__main__':

    try: # 尋找版本
        version  = find_FinalVersion('train')

    except FileNotFoundError: # 訓練第一代
        print('目前訓練為第1代')
        model = YOLO('./models/yolov8n.pt')
        results = model.train(data="data.yaml", epochs=300, batch=3)
        results = model.val(data="data.yaml")

        if not path.exists(f'./models/G1'):
            mkdir(f'./models/G1')
        copy(f'./runs/detect/train/weights/best.pt',
             f'./models/G1')
        copy(f'./runs/detect/train/weights/last.pt',
             f'./models/G1')
        
    except Exception as e:  # 例外狀況
        print(e)

    else: # 訓練新一代
        if version == '':
            version = 1
        else:
            version = int(version)
        print(f'目前訓練為第{version+1}代')

        model = YOLO(f'./models/G{version}/last.pt')
        results = model.train(data="data.yaml", epochs=300, batch=3, device='0')
        results = model.val(data="data.yaml")

        if not path.exists(f'./models/G{version+1}'):
            mkdir(f'./models/G{version+1}')
        copy(f'./runs/detect/train{version+1}/weights/best.pt',
             f'./models/G{version+1}')
        copy(f'./runs/detect/train{version+1}/weights/last.pt',
             f'./models/G{version+1}')
        
    finally:
        exit()