from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense, BatchNormalization, Input


def ConvBlock(x, filter) :
    x = Conv2D(filter, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPool2D()(x)
    return x

'''
    밑바닥부터 시작하는 모델
'''
def DogAndCat() :
    inputs = Input(shape=(224, 224, 3))
    x = ConvBlock(inputs, 32)
    x = ConvBlock(x, 64)
    x = ConvBlock(x, 128)
    x = ConvBlock(x, 256)

    x = Conv2D(512, kernel_size=3, padding='same', activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, x)

    return model

'''
    학습이 잘되는 모델 (기존 ResNet50 떼와서 학습하기)
'''
# def DogAndCat() :
#     inputs = Input(shape=(224, 224, 3))
#     x = ResNet50(include_top=False, weights='imagenet', pooling = 'avg', input_shape=(224, 224, 3))(inputs)
#     x = Dense(128, activation="relu")(x)
#     #x = Dense(2, activation="softmax")(x)
#     x = Dense(1, activation="sigmoid")(x)
#     model = Model(inputs, x)
#     return model