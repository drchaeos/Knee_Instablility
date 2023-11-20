## 필요한 모듈 import

import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import csv
import math

import torch
from typing import Tuple, List, Sequence, Callable, Dict

from detectron2.structures import BoxMode
from detectron2.engine import HookBase
from detectron2.utils.events import get_event_storage
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog

## parameter 설정
dataname = '[ACL]'
column_path = './columns/knee_stress.csv'
test_dir = './test_imgs'         # test image 들어간 directory
model_dir = './model'                      # model.pth 들어간 directory
output_dir = './out'                     # output파일들 저장될 directory
num_keypoints = 6
keypoint_names = {0: 'FP', 1: 'TP', 2: 'MTP1', 3: 'MTP2', 4: 'B1', 5: 'B2'}
edges = [(2, 3), (4, 5)]

## 함수 정의
def draw_keypoints(image, keypoints,
                   edges: List[Tuple[int, int]] = None,
                   keypoint_names: Dict[int, str] = None,
                   boxes: bool = True) -> None:

    keypoints = keypoints.astype(np.int64)
    keypoints_ = keypoints.copy()
    np.random.seed(42)
    colors = {k: tuple(map(int, np.random.randint(0, 255, 3))) for k in range(num_keypoints)}
    if len(keypoints_) == (2 * num_keypoints):
        keypoints_ = [[keypoints_[i], keypoints_[i + 1]] for i in range(0, len(keypoints_), 2)]

    assert isinstance(image, np.ndarray), "image argument does not numpy array."
    image_ = np.copy(image)
    for i, keypoint in enumerate(keypoints_):
        cv2.circle(
            image_,
            tuple(keypoint),
            3, colors.get(i), thickness=3, lineType=cv2.FILLED)

        if keypoint_names is not None:
            cv2.putText(
                image_,
                f'{i}: {keypoint_names[i]}',
                tuple(keypoint),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if edges is not None:
        for i, edge in enumerate(edges):
            cv2.line(
                image_,
                tuple(keypoints_[edge[0]]),
                tuple(keypoints_[edge[1]]),
                colors.get(edge[0]), 20, lineType=cv2.LINE_AA)
    if boxes:
        x1, y1 = min(np.array(keypoints_)[:, 0]), min(np.array(keypoints_)[:, 1])
        x2, y2 = max(np.array(keypoints_)[:, 0]), max(np.array(keypoints_)[:, 1])
        cv2.rectangle(image_, (x1, y1), (x2, y2), (255, 100, 91), thickness=3)

    # 모든 점 x,y 좌표 추출
    FP_x = keypoints_[0][0]
    FP_y = keypoints_[0][1]
    TP_x = keypoints_[1][0]
    TP_y = keypoints_[1][1]
    MTP1_x = keypoints_[2][0]
    MTP1_y = keypoints_[2][1]
    MTP2_x = keypoints_[3][0]
    MTP2_y = keypoints_[3][1]
    B1_x = keypoints_[4][0]
    B1_y = keypoints_[4][1]
    B2_x = keypoints_[5][0]
    B2_y = keypoints_[5][1]

    # medial tibial plateau line에 수직인 선(위, 아래 마진 500)
    m = (MTP1_y - MTP2_y) / (MTP1_x - MTP2_x)
    if m != 0:
        n = -(1/m)
        a1 = FP_y - (n * FP_x)
        x1_min = (500 - a1) / n
        x1_max = ((image_.shape[1] - 200) - a1) / n

        a2 = TP_y - (n * TP_x)
        x2_min = (500 - a2) / n
        x2_max = ((image_.shape[1] - 200) - a2) / n

    if  m == 0:
        x1_min = FP_x
        x1_max = FP_x
        x2_min = TP_x
        x2_max = TP_x

    cv2.line(image_, (int(x1_min), 500), (int(x1_max), (image_.shape[1] - 200)), (128, 0, 0), 20, lineType=cv2.LINE_AA)
    cv2.line(image_, (int(x2_min), 500), (int(x2_max), (image_.shape[1] - 200)), (0, 0, 128), 20, lineType=cv2.LINE_AA)

    # B1과 B2 거리 구하기, mm 비율 구하기
    global d1, d, dmm
    p1 = abs(B1_x - B2_x)
    p2 = abs(B1_y - B2_y)
    d1 = math.sqrt((p1 * p1) + (p2 * p2))
    lmm = (d1 / 43)
    # cv2.rectangle(image_, (0, 0), (600, 200), (255, 255, 255), -1)
    # cv2.putText(image_, '1cm = ' + str(lcm), (50, 50), 4, 1, (0, 0, 0), 1, cv2.LINE_AA)

    # FP-TP 간 거리 구하기
    a3 = TP_y - (m * TP_x)
    x3 = (a1 - a3) / (m - n)
    y3 = (m * x3) + a3
    p3 = abs(x3 - TP_x)
    p4 = abs(y3 - TP_y)
    d = math.sqrt((p3 * p3) + (p4 * p4))
    if lmm != 0:
        dmm = d / lmm
    else:
        dmm = 0

    # cv2.putText(image_, 'D = ' + str(dcm) + 'cm', (50, 150), 4, 1, (0, 0, 0), 1, cv2.LINE_AA)

    return image_


def save_samples(dst_path, image_path, csv_path, mode="random", size=None, index=None):
    df = pd.read_csv(csv_path)
    filename = df.iloc[0, 0]
    filename = os.path.splitext(filename)
    filename = filename[0]
    # csv 파일로 저장
    output_file = open(os.path.join(dst_path, f'{dataname}_result_{filename}.csv'), 'w', newline='')
    f = csv.writer(output_file)
    # csv 파일에 header 추가
    f.writerow(["image", "B1-B2 distance", "B1-B2 length", "FP-TP distance", "FP-TP length"])

    if mode == "random":
        assert size is not None, "mode argument is random, but size argument is not given."
        choice_idx = np.random.choice(len(df), size=size, replace=False)
    if mode == "choice":
        assert index is not None, "mode argument is choice, but index argument is not given."
        choice_idx = index

    for idx in choice_idx:
        image_name = df.iloc[idx, 0]
        keypoints = df.iloc[idx, 1:]
        image = cv2.imread(os.path.join(image_path, image_name), cv2.IMREAD_COLOR)

        combined = draw_keypoints(image, keypoints, edges, keypoint_names, boxes=False)
        cv2.imwrite(os.path.join(dst_path, "result_" + image_name), combined)
        f.writerow([image_name, format(d1, ".3f"), '43mm', format(d, ".3f"), format(dmm, ".3f") + 'mm'])

## inference
cfg = get_cfg()
cfg.MODEL.DEVICE = "cpu"
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATALOADER.NUM_WORKERS = 0  # On Windows environment, this value must be 0.
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.RETINANET.NUM_CLASSES = 1
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = num_keypoints
cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((num_keypoints, 1), dtype=float).tolist()
cfg.OUTPUT_DIR = output_dir

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

cfg.MODEL.WEIGHTS = os.path.join(model_dir, "ACL120_model.pth")  # 학습된 모델 들어가 있는 곳
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # custom testing threshold
predictor = DefaultPredictor(cfg)

test_list = os.listdir(test_dir)
test_list.sort()
except_list = []

files = []
preds = []
for file in tqdm(test_list):
    filepath = os.path.join(test_dir, file)
    im = cv2.imread(filepath)
    outputs = predictor(im)
    outputs = outputs["instances"].to("cpu").get("pred_keypoints").numpy()
    files.append(file)
    pred = []
    try:
        for out in outputs[0]:
            pred.extend([float(e) for e in out[:2]])
    except IndexError:
        pred.extend([0] * (2 * num_keypoints))
        except_list.append(filepath)
    preds.append(pred)

df_sub = pd.read_csv(column_path)
df = pd.DataFrame(columns=df_sub.columns)
df["image"] = files
df.iloc[:, 1:] = preds

df.to_csv(os.path.join(cfg.OUTPUT_DIR, f"{dataname}_keypoints.csv"), index=False)
if except_list:
    print(
        "The following images are not detected keypoints. The row corresponding that images names would be filled with 0 value."
    )
    print(*except_list)


save_samples(cfg.OUTPUT_DIR, test_dir, os.path.join(cfg.OUTPUT_DIR, f"{dataname}_keypoints.csv"), mode="choice", size=5, index=range(len(files)))





