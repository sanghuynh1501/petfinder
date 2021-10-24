import pandas as pd

train0 = pd.read_csv('petfinder-pawpularity-score/train.csv')
train1 = pd.read_csv('petfinder-adoption-prediction/train/train.csv')

for ID, score in zip(train0['ID'], train0['Pawpularity']):
    print(ID, score)