from Dataloader import Dataloader
from Model_class import DogAndCatModel
from Model_func import DogAndCat

test_path = "D:/IPCV/dogs_vs_cats/dogs-cats-images/dataset/test_set"
training_path = "D:/IPCV/dogs_vs_cats/dogs-cats-images/dataset/training_set"

train_generator = Dataloader(training_path, 32)
# Class
#model = DogAndCatModel()
#functional API
model = DogAndCat()
model.summary()
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['binary_accuracy'])

model.fit(train_generator, epochs=5)

test_generator = Dataloader(test_path, 32)
model.evaluate(test_generator)
