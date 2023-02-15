import numpy as np
import os
import torch
import pandas as pd
import cv2

class fake_and_real(object):
    def __init__(self, root_path='data_train.csv', image_size=(112, 112)):
         self.data_path = pd.read_csv(root_path)    
         self.num_images = len(self.data_path)      
         self.image_size = image_size               

    
    def __getitem__(self, index):
        # read image path from cvs file
        image_path = os.path.join(self.data_path.iloc[index, 0])

        # read image from image path
        img = cv2.imread(image_path)

        # Convert image from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize all image into same size 112x112
        img = cv2.resize(img, self.image_size)

         # read label to train cross_entropy loss
        label_cross = self.data_path.iloc[index, 1]

        # if image is gray image --> stack gray image 3 times
        # make gray image become 3 channels
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)

        # Augmentation using flip technique
        flip = np.random.choice(2)*2-1
        img = img[:,::flip, :]
        img = img/255

        # Convert image from format [h, w, channel] into format [channel, h, w]
        img = img.transpose(2, 0, 1)

        # Convert image from array to torch
        img = torch.from_numpy(img).float()

        return img, torch.from_numpy(np.array(label_cross, dtype=np.long))

    def __len__(self):
        return self.num_images