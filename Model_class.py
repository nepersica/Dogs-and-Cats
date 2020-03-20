from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense, BatchNormalization

'''
    모델 본문
'''
class DogAndCatModel(Model):
    def __init__(self):
        super(DogAndCatModel, self).__init__()

    def build(self, input_shape):
        self.Conv1 = Conv2D(32, kernel_size=3, padding='same', activation='relu')
        self.bn1 = BatchNormalization(axis=-1)
        self.pool1 = MaxPool2D()

        self.Conv2 = Conv2D(64, kernel_size=3, padding='same', activation='relu')
        self.bn2 = BatchNormalization(axis=-1)
        self.pool2 = MaxPool2D()

        self.Conv3 = Conv2D(128, kernel_size=3, padding='same', activation='relu')
        self.bn3 = BatchNormalization(axis=-1)
        self.pool3 = MaxPool2D()

        self.Conv4 = Conv2D(256, kernel_size=3, padding='same', activation='relu')
        self.bn4 = BatchNormalization(axis=-1)
        self.pool4 = MaxPool2D()

        self.Conv5 = Conv2D(512, kernel_size=3, padding='same', activation='relu')

        self.GAP = GlobalAveragePooling2D()

        self.dense = Dense(1, activation="sigmoid")

    def call(self, x):
        x = self.Conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)

        x = self.Conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)

        x = self.Conv3(x)
        x = self.bn3(x)
        x = self.pool3(x)

        x = self.Conv4(x)
        x = self.bn4(x)
        x = self.pool4(x)

        x = self.Conv5(x)

        x = self.GAP(x)
        x = self.dense(x)
        return x




