import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.preprocessing.image import ImageDataGenerator

# Define the directory where the model is saved
model_dir = 'D:/Final year project/model'

# Load the trained model
model_path = os.path.join(model_dir, 'model-2.h5')
model = tf.keras.models.load_model(model_path)

# Set up ImageDataGenerator for test data
test_datagen = ImageDataGenerator(rescale=1./255)  # Rescale pixel values to [0, 1]

test_generator = test_datagen.flow_from_directory(
    'D:/Final year project/pre-processed_dataset/test',
    target_size=(250, 250),  # Set the target size to match the input size of your model
    batch_size=35,  # Set the batch size according to your training configuration
    class_mode='categorical',  # Assuming you're using categorical labels
    color_mode='grayscale',  # Set the color mode according to your model input
    shuffle=False  # Ensure that predictions are ordered correctly
)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_generator, steps=len(test_generator))
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

# Make predictions on the test data
predictions = model.predict(test_generator, steps=len(test_generator))

# Get true labels
true_labels = test_generator.classes
class_indices = test_generator.class_indices

# Get predicted labels
predicted_labels = np.argmax(predictions, axis=1)

# Compute confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Display confusion matrix
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=class_indices.keys())
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Plot some sample images with their true and predicted labels
num_samples = 20
sample_indices = np.random.choice(len(true_labels), num_samples, replace=False)

plt.figure(figsize=(15, 10))
num_cols = 7
num_rows = num_samples // num_cols + (num_samples % num_cols > 0)
for i, index in enumerate(sample_indices, 1):
    plt.subplot(num_rows, num_cols, i)
    img = plt.imread(test_generator.filepaths[index])
    plt.imshow(img, cmap='gray')
    true_label = list(test_generator.class_indices.keys())[true_labels[index]]
    pred_label = list(test_generator.class_indices.keys())[predicted_labels[index]]
    plt.title(f'True: {true_label}\nPredicted: {pred_label}')
    plt.axis('off')
plt.tight_layout(h_pad=2, w_pad=2)
plt.show()
