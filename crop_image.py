import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from imgaug import augmenters as iaa

IMAGE_SIZE = 384
train = pd.read_csv('petfinder-pawpularity-score/train_yolo.csv')

# for file_path, x_min, x_max, y_min, y_max in zip(train['file_path'], train['x_min'], train['x_max'], train['y_min'], train['y_max']):
#     x_min, x_max, y_min, y_max = round(x_max), round(x_min), round(y_min), round(y_max)
#     image = cv2.imread(file_path)
#     print(file_path, x_min, x_max, y_min, y_max)
#     print('image.shape ', image.shape)
#     image = image[x_min:x_max,y_min:y_max]
#     print('image.shape ', image.shape)
#     cv2.imshow('image ', image)
#     cv2.waitKey(0)

for file_path, coords in zip(train['file_path'], train['coords']):
    coords = coords.replace('[(', '').replace(')]', '')
    coords = [cord.split(', ') for cord in coords.split('), (')]
    image = cv2.imread(file_path)
    
    x1s = []
    y1s = []
    x2s = []
    y2s = []

    folder = file_path.split('.')[0]
    folder = folder.replace('train', 'train_crop_large')
    for i in range(0, 20, 1):
        new_folder = f'{folder}_{i}'
        os.makedirs(new_folder)
        for ord, coord in enumerate(coords):
            if coord[0] != '[]':
                coord = [round(float(cord)) for cord in coord]
                new_image = image[coord[1]:coord[3], coord[0]:coord[2]]
            else:
                new_image = image
            if i < 6:
                au_image = new_image
            else:
                new_image = np.expand_dims(new_image, 0)
                seq = iaa.Sequential([
                    iaa.Crop(px=(10, 50)),
                    iaa.Affine(rotate=(-10, 10)),
                    iaa.Fliplr(0.5)
                ])
                au_image = seq(images=new_image)
                au_image = au_image[0]
            au_image = cv2.resize(au_image, (IMAGE_SIZE, IMAGE_SIZE))
            cv2.imwrite(f'{new_folder}/image_{ord}.png', au_image)


    # for coord in coords:
    #     if coord[0] != '[]':
    #         coord = [round(float(cord)) for cord in coord]
    #         # image = image[coord0[1]:coord0[3],coord0[0]:coord0[2]]
    #         image = cv2.rectangle(image, (coord[0], coord[1]), (coord[2], coord[3]), (255, 0, 0), 1)
    #         x1s.append(coord[0])
    #         y1s.append(coord[1])
    #         x2s.append(coord[2])
    #         y2s.append(coord[3])
    
    # cv2.imshow('image ', image)

    # if len(x1s) > 0 and len(y1s) > 0 and len(x2s) > 0 and len(y2s) > 0:
    #     x_min = np.min(x1s)
    #     y_min = np.min(y1s)
    #     x_max = np.max(x2s)
    #     y_max = np.max(y2s)

    #     # if coords[0][0] != '[]':
    #     #     coord0 = [round(float(cord)) for cord in coords[0]]
    #     #     print('coord0 ', coord0)
    #     # print(file_path, x_min, x_max, y_min, y_max)
    #     # print('image.shape ', image.shape)
    #     image = image[y_min:y_max,x_min:x_max]
    #     # print('image.shape ', image.shape)
    #     cv2.imshow('image_crop', image)
    cv2.waitKey(0)