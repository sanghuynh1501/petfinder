# Author: Zylo117

"""
Simple Inference Script of EfficientDet-Pytorch
"""
import time
import torch
from torch.backends import cudnn
from matplotlib import colors

# from backbone import EfficientDetBackbone
import cv2
import numpy as np
import pandas as pd
from efficientdet.backbone import EfficientDetBackbone
from efficientdet.efficientdet.utils import BBoxTransform, ClipBoxes
from efficientdet.utils.utils import STANDARD_COLORS, get_index_label, invert_affine, plot_one_box, postprocess, preprocess, standard_to_bgr

# from efficientdet.utils import BBoxTransform, ClipBoxes
# from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

compound_coef = 7
force_input_size = None  # set None to use default size
img_path = 'petfinder-pawpularity-score/train/175750d5b21722b7451361df8852e374.jpg'

# image = cv2.imread(img_path)
# cv2.imshow('image ', image)
# cv2.waitKey(0)

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.2
iou_threshold = 0.2

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']

select_labels = [obj_list.index('cat'), obj_list.index('dog'), obj_list.index('bear'), obj_list.index('teddy bear'), obj_list.index('sheep')]

color_list = standard_to_bgr(STANDARD_COLORS)
# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             ratios=anchor_ratios, scales=anchor_scales)
model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth', map_location='cpu'))
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

def get_pet_number(img_path):
    img = cv2.imread(img_path)
    _, framed_imgs, _ = preprocess(img, max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    with torch.no_grad():
        _, regression, classification, anchors = model(x)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)
        return len(list(filter(lambda number: number in select_labels, out[0]['class_ids'])))

def get_crop_info(image):
    ori_imgs, framed_imgs, framed_metas = preprocess(image, max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    with torch.no_grad():
        _, regression, classification, anchors = model(x)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)

    out = invert_affine(framed_metas, out)
    coords = []
    for i in range(len(out[0]['rois'])):
        if out[0]['class_ids'][i] in select_labels:
            x1, y1, x2, y2 = out[0]['rois'][i].astype(np.int)
            coords.append(tuple([x1, y1, x2, y2]))

    return ori_imgs[0], coords

# get_crop_info('petfinder-pawpularity-score/train/0a0da090aa9f0342444a7df4dc250c66.jpg')
data = pd.read_csv('petfinder-pawpularity-score/train_yolo.csv')
data['n_pets_new'] = data['file_path'].apply(get_pet_number)
data.to_csv('petfinder-pawpularity-score/train_yolo_new.csv', sep=',')