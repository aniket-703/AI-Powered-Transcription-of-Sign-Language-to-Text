# Importing the Keras libraries and packages
import os
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator

# Image size
sz = 250

# Step 1 - Building the CNN
# Initializing the CNN
classifier = Sequential()

# First convolution layer and pooling
classifier.add(Convolution2D(128, (3, 3), input_shape=(sz, sz, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.4))

# Second convolution layer and pooling
classifier.add(Convolution2D(256, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.4))

# Third convolution layer and pooling
classifier.add(Convolution2D(512, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.4))


# Flattening the layers
classifier.add(Flatten())

# Adding a fully connected layer
classifier.add(Dense(units=512, activation='relu'))
classifier.add(Dropout(0.40))
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=256, activation='relu'))
classifier.add(Dropout(0.3))
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dropout(0.2))

# Output layer for multi-class classification
classifier.add(Dense(units=37, activation='softmax')) # softmax for more than 2

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # categorical_crossentropy for more than 2

# Display model summary
classifier.summary()


# Step 2 - Preparing the train/test data and training the model
# Code copied from - https://keras.io/preprocessing/image/

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('D:/Final year project/pre-processed_dataset/train',
                                                 target_size=(sz, sz),
                                                 batch_size=30,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('D:/Final year project/pre-processed_dataset/test',
                                            target_size=(sz, sz),
                                            batch_size=30,
                                            color_mode='grayscale',
                                            class_mode='categorical') 


class_names = list(training_set.class_indices.keys())
print(class_names)

# Fitting the model
classifier.fit_generator(
        training_set,
        steps_per_epoch=len(training_set),
        epochs=30,
        validation_data=test_set,
        validation_steps=len(test_set))


# Define the directory where you want to save the model
save_dir = 'D:/Final year project/model'

# Saving the model architecture
model_json = classifier.to_json()
with open(os.path.join(save_dir, "model-1st.json"), "w") as json_file:
    json_file.write(model_json)
print('Model architecture Saved')

# Saving the entire model
classifier.save(os.path.join(save_dir, 'model-1st.h5'))
print('Entire model Saved')
