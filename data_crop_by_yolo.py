import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from multiprocessing import cpu_count

import sys
import imageio
import warnings

import torch
import tensorflow as tf

# Ignore Warnings
warnings.filterwarnings("ignore")

print(f'tensorflow version: {tf.__version__}')
print(f'tensorflow keras version: {tf.keras.__version__}')
print(f'python version: P{sys.version}')

# Efficient Data Types
dtype = {
    'Id': 'string',
    'Subject Focus': np.uint8, 'Eyes': np.uint8, 'Face': np.uint8, 'Near': np.uint8,
    'Action': np.uint8, 'Accessory': np.uint8, 'Group': np.uint8, 'Collage': np.uint8,
    'Human': np.uint8, 'Occlusion': np.uint8, 'Info': np.uint8, 'Blur': np.uint8,
    'Pawpularity': np.uint8,
}

train = pd.read_csv('petfinder-pawpularity-score/train.csv', dtype=dtype)
test = pd.read_csv('petfinder-pawpularity-score/test.csv', dtype=dtype)

def get_image_file_path(image_id):
    return f'petfinder-pawpularity-score/train/{image_id}.jpg'

train['file_path'] = train['Id'].apply(get_image_file_path)

yolov5x6_model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')

# Get Image Info
def get_image_info(file_path, plot=False):
    # Read Image
    image = imageio.imread(file_path)
    
    if plot: # Debug Plots
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(image)
        
    # Get YOLOV5 results
    results = yolov5x6_model(image)
    
    # Dictionary to Save Image Info
    h, w, _ = image.shape
    image_info = { 
        'n_pets': 0, # Number of pets in the image
        'labels': [], # Label assigned to found objects
        'thresholds': [], # confidence score
        'coords': [], # coordinates of bounding boxes
        'x_min': 0, # minimum x coordinate of pet bounding box
        'x_max': w - 1, # maximum x coordinate of pet bounding box
        'y_min': 0, # minimum y coordinate of pet bounding box
        'y_max': h - 1, # maximum x coordinate of pet bounding box
    }
    
    # Save info for each pet
    for x1, y1, x2, y2, treshold, label in results.xyxy[0].cpu().detach().numpy():
        label = results.names[int(label)]
        if label in ['cat', 'dog']:
            image_info['n_pets'] += 1
            image_info['labels'].append(label)
            image_info['thresholds'].append(treshold)
            image_info['coords'].append(tuple([x1, y1, x2, y2]))
            image_info['x_min'] = max(x1, image_info['x_min'])
            image_info['x_max'] = min(x2, image_info['x_max'])
            image_info['y_min'] = max(y1, image_info['y_min'])
            image_info['y_max'] = min(y2, image_info['y_max'])

            if plot:
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
                # Add the patch to the Axes
                ax.add_patch(rect)

    if plot:
        plt.show()
        
    return image_info

# Image Info
IMAGES_INFO = {
    'n_pets': [],
    'label': [],
    'coords': [],
    'x_min': [],
    'x_max': [],
    'y_min': [],
    'y_max': [],
}

for idx, file_path in enumerate(train['file_path']):
    image_info = get_image_info(file_path, plot=False)
    
    IMAGES_INFO['n_pets'].append(image_info['n_pets'])
    IMAGES_INFO['coords'].append(image_info['coords'])
    IMAGES_INFO['x_min'].append(image_info['x_min'])
    IMAGES_INFO['x_max'].append(image_info['x_max'])
    IMAGES_INFO['y_min'].append(image_info['y_min'])
    IMAGES_INFO['y_max'].append(image_info['y_max'])
    
    # Not Every Image can be Correctly Classified
    labels = image_info['labels']
    if len(set(labels)) == 1: # unanimous label
        IMAGES_INFO['label'].append(labels[0])
    elif len(set(labels)) > 1: # Get label with highest confidence
        IMAGES_INFO['label'].append(labels[0])
    else: # unknown label, yolo could not find pet
        IMAGES_INFO['label'].append('unknown')

train['n_pets'] = IMAGES_INFO['n_pets']
train['coords'] = IMAGES_INFO['coords']
train['label'] = IMAGES_INFO['label']
train['x_min'] = IMAGES_INFO['x_min']
train['x_max'] = IMAGES_INFO['x_max']
train['y_min'] = IMAGES_INFO['y_min']
train['y_max'] = IMAGES_INFO['y_max']

train.to_csv('petfinder-pawpularity-score/train_yolo.csv', sep=',')