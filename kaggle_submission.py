import os
import torch
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from imgaug import augmenters as iaa
from swintransformer.model import SwinTransformer

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass


IMAGE_SIZE = 224
EIMAGE_SIZE = 224
D_MODEL = 32
DFF = 64

# Efficient Data Types
dtype = {
    'Id': 'string',
    'Subject Focus': np.uint8, 'Eyes': np.uint8, 'Face': np.uint8, 'Near': np.uint8,
    'Action': np.uint8, 'Accessory': np.uint8, 'Group': np.uint8, 'Collage': np.uint8,
    'Human': np.uint8, 'Occlusion': np.uint8, 'Info': np.uint8, 'Blur': np.uint8,
    'Pawpularity': np.uint8,
}

test = pd.read_csv('AdoptionSpeed.csv', dtype=dtype)


def get_image_file_path(image_id):
    return f'petfinder-pawpularity-score/train/{image_id}.jpg'


test['file_path'] = test['Id'].apply(get_image_file_path)

yolov5x6_model = torch.hub.load(
    'ultralytics/yolov5', 'yolov5x6', pretrained=True)
swin_model = SwinTransformer(
    'swin_large_224', num_classes=1000, include_top=False, pretrained='swintransformer/swin_large_224')
efficient = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)


def image_iaa(image):
    image = np.expand_dims(image, 0)
    seq = iaa.Sequential([
        iaa.Crop(px=(10, 50)),
        iaa.Affine(rotate=(-10, 10)),
        iaa.Fliplr(0.5),
        iaa.MultiplyAndAddToBrightness(mul=(0.9, 1.1), add=(-30, 30))
    ])
    au_image = seq(images=image)
    au_image = au_image[0]

    return au_image


def get_image_info(file_path):
    image = cv2.imread(file_path)

    results = yolov5x6_model(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    coords = []
    max_score = 0
    labels = []
    all_coords = []
    for x1, y1, x2, y2, score, label in results.xyxy[0].cpu().detach().numpy():
        label = results.names[int(label)]
        if label in ['cat', 'dog', 'teddy bear', 'cow', 'horse', 'bear', 'bird', 'zebra']:
            if score > max_score:
                coords = [tuple([x1, y1, x2, y2])]
                max_score = score
        labels.append(label)
        all_coords.append(tuple([x1, y1, x2, y2]))

    return image, coords, labels, all_coords


def extract_feature(image):
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = tf.keras.applications.imagenet_utils.preprocess_input(
        image, mode='torch')
    image = np.expand_dims(image, 0)
    feature = swin_model(image)
    return feature


def extract_feature_efficient(image):
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    image = np.expand_dims(image, 0)
    feature = efficient(image)
    return feature


def render_target(target):
    result = []
    max_length = len(target)
    for i in range(len(target)):
        if float(target[i]) == 1:
            result.append(float(i))
    length = len(result)
    return np.concatenate([np.array(result), np.zeros((max_length - length,))])


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)

predict_scores = []
real_scores = []

with tqdm(total=len(test.index)) as pbar:
    for file_path in test['file_path']:
        image, coords, labels, all_coords = get_image_info(file_path)

        if len(coords) > 0:
            coord_1_min = float('inf')
            coord_3_max = 0
            coord_0_min = float('inf')
            coord_2_max = 0
            for ord, coord in enumerate(coords):
                coord = [round(float(cord)) for cord in coord]
                if coord[1] < coord_1_min:
                    coord_1_min = coord[1]
                if coord[3] > coord_3_max:
                    coord_3_max = coord[3]
                if coord[0] < coord_0_min:
                    coord_0_min = coord[0]
                if coord[2] > coord_2_max:
                    coord_2_max = coord[2]
            new_image = image[coord_1_min:coord_3_max,
                              coord_0_min:coord_2_max]
            for i in range(6):
                if i >= 2:
                    new_image = image_iaa(new_image)
                features = []
                feature = extract_feature(new_image)
                efeature = extract_feature_efficient(new_image)
                features.append(feature)
                features.append(efeature)

                file_path_new = file_path.replace(
                    'train', 'feature_full_large_new_new').replace('.jpg', '')
                if not os.path.isdir(f'{file_path_new}_{i}'):
                    os.makedirs(f'{file_path_new}_{i}')
                for idx, feature in enumerate(features):
                    np.save(f'{file_path_new}_{i}/feature_{idx}.npy', feature)
        else:
            new_image = None
            if len(all_coords):
                coord_1_min = float('inf')
                coord_3_max = 0
                coord_0_min = float('inf')
                coord_2_max = 0
                for ord, coord in enumerate(all_coords):
                    coord = [round(float(cord)) for cord in coord]
                    if coord[1] < coord_1_min:
                        coord_1_min = coord[1]
                    if coord[3] > coord_3_max:
                        coord_3_max = coord[3]
                    if coord[0] < coord_0_min:
                        coord_0_min = coord[0]
                    if coord[2] > coord_2_max:
                        coord_2_max = coord[2]
                image = image[coord_1_min:coord_3_max,
                              coord_0_min:coord_2_max]
                new_image = image
            else:
                new_image = image
            for i in range(6):
                if i >= 2:
                    new_image = image_iaa(new_image)
                features = []
                feature = extract_feature(new_image)
                efeature = extract_feature_efficient(new_image)
                features.append(feature)
                features.append(efeature)

                file_path_new = file_path.replace(
                    'train', 'feature_full_large_new_new').replace('.jpg', '')
                if not os.path.isdir(f'{file_path_new}_{i}'):
                    os.makedirs(f'{file_path_new}_{i}')
                for idx, feature in enumerate(features):
                    np.save(f'{file_path_new}_{i}/feature_{idx}.npy', feature)
        pbar.update(1)
