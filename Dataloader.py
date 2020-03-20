from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras import utils
import math
import numpy as np
import os


class Dataloader(utils.Sequence):
    def __init__(self, folder_path, batch_size, shuffle=True):
        self.image_list, self.lable_list = self.GetDataList(folder_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.image_list) / self.batch_size)

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = [self.image_list[k] for k in indexes]
        batch_y = [self.lable_list[k] for k in indexes]

        # return np.array([batch_x]), np.array(batch_y)

        return np.array([resize(imread(file_name), (224, 224)) / 255. for file_name in batch_x]), np.array(batch_y)
    # image의 크기를 224x224로 변경하고 배열 안에 넣어줌으로써 batch 형태인 1x224x224x3으로 바꿔줌.
    # 그리고 255로 나누어 양자화 시켜주었음.

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def GetDataList(self, folder_path: str):
        train_list = []
        lable_list = []

        lables = ["dogs", "cats"]
        for lable in lables:
            image_names = os.listdir("{}/{}".format(folder_path, lable))

            for image_name in image_names:
                image_path = "{}/{}/{}".format(folder_path, lable, image_name)
                train_list.append(image_path)
                if lable == "dogs":
                    lable_list.append(0)
                else:
                    lable_list.append(1)

        return train_list, lable_list

# dataLoader = Dataloader("./train", 2)
# dataLoader.on_epoch_end()
# print(dataLoader.__getitem__(0))
# print(dataLoader.__getitem__(1))