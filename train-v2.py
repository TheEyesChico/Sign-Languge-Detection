# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
import os

import matplotlib.pyplot as plt

sz = 128
# Step 1 - Building the CNN

# Initializing the CNN
classifier = Sequential()

classifier.add(Convolution2D(32, (7, 7), input_shape=(sz, sz, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(32, (5, 5), input_shape=(sz, sz, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Second convolution layer and pooling
classifier.add(Convolution2D(32, (5, 5), input_shape=(sz, sz, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# classifier.add(Convolution2D(128, (3, 3), input_shape=(sz, sz, 1), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers
classifier.add(Flatten())

# Adding a fully connected layer
classifier.add(Dense(units=512, activation='relu'))
classifier.add(Dropout(0.3))
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dense(units=27, activation='softmax')) # softmax for more than 2

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # categorical_crossentropy for more than 2

# Step 2 - Preparing the train/test data and training the model
classifier.summary()
# Code copied from - https://keras.io/preprocessing/image/
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('Threshold/All_Gestures/train',
                                                 target_size=(sz, sz),
                                                 batch_size=64,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('Threshold/All_Gestures/val',
                                            target_size=(sz , sz),
                                            batch_size=64,
                                            color_mode='grayscale',
                                            class_mode='categorical')

r = classifier.fit_generator(
        training_set,
        steps_per_epoch=len(training_set), # No of images in training set
        epochs=10,
        validation_data=test_set,
        validation_steps=len(test_set))# No of images in test set

model_version = "v4_epoch-10"
if not os.path.exists(model_version):
    os.makedirs(model_version)

# Saving the model
model_json = classifier.to_json()
with open(model_version+"/model-bw.json", "w") as json_file:
    json_file.write(model_json)
print('Model Saved')
classifier.save_weights(model_version+"/model-bw-weights.h5")
print('Weights saved')
classifier.save(model_version+"/model.h5")
print('Model saved')

# summarize history for accuracy
plt.plot(r.history['acc'],label='train accuracy')
plt.plot(r.history['val_acc'],label='validation accuracy')
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.savefig(model_version+'/Accuracy.png')
plt.show()
# summarize history for loss
plt.plot(r.history['loss'],label="train loss")
plt.plot(r.history['val_loss'],label="validation loss")
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.savefig(model_version+'/Loss.png')
plt.show()