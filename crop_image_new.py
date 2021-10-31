import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

train = pd.read_csv('petfinder-adoption-prediction/train/train.csv')

with tqdm(total=len(train.index)) as pbar:
    for PetID, photo_count in zip(train['PetID'], train['PhotoAmt']):
        max_height = 0
        max_width = 0
        if 5 <= int(photo_count) <= 15:
            images = []
            for count in range(int(photo_count)):
                path = f'petfinder-adoption-prediction/train_images/{PetID}-{count + 1}.jpg'
                image = cv2.imread(path)
                if image.shape[0] > max_height:
                    max_height = image.shape[0]
                if image.shape[1] > max_width:
                    max_width = image.shape[1]
                images.append(image)
            all_image = None
            if max_height > max_width:
                for idx, image in enumerate(images):
                    pad_image = np.ones((max_height - image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255
                    image = np.concatenate([image, pad_image], 0)
                    if all_image is None:
                        all_image = image
                    else:
                        all_image = np.concatenate([all_image, image], 1)
            else:
                for idx, image in enumerate(images):
                    pad_image = np.ones((image.shape[0], max_width - image.shape[1], 3), dtype=np.uint8) * 255
                    image = np.concatenate([image, pad_image], 1)
                    if all_image is None:
                        all_image = image
                    else:
                        all_image = np.concatenate([all_image, image], 0)
            
            cv2.imwrite(f'petfinder-adoption-prediction/train_crop_images/{PetID}.jpg', all_image)
        pbar.update(1)