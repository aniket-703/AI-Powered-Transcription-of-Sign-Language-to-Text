import cv2
import random
import os
from shutil import copyfile
from sklearn.model_selection import train_test_split

# Function to preprocess an image
def preprocess_image(image):
    
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return grayscale_image

# Path to the original dataset directory
original_dataset_dir = 'D:/Final year project/original_images'

# Path to directory where processed dataset will be saved
processed_dir = 'D:Final year project/pre-processed_dataset'
os.makedirs(processed_dir, exist_ok=True)

# Path to directory where preprocessed train images will be saved
train_dir = os.path.join(processed_dir, 'train')
os.makedirs(train_dir, exist_ok=True)

# Path to directory where preprocessed test images will be saved
test_dir = os.path.join(processed_dir, 'test')
os.makedirs(test_dir, exist_ok=True)

# Lists to hold paths of train and test images
train_image_paths = []
test_image_paths = []

# Define the train-test split ratio
train_test_split_ratio = 0.8  # 80% train, 20% test

# Get the total number of images
total_images = sum(len(files) for _, _, files in os.walk(original_dataset_dir))

# Counter to keep track of processed images
processed_images = 0

# Loop through each image in the original dataset directory
for class_dir in os.listdir(original_dataset_dir):
    class_dir_path = os.path.join(original_dataset_dir, class_dir)
    if os.path.isdir(class_dir_path):
        # Create a directory in train and test for each class
        train_class_dir = os.path.join(train_dir, class_dir)
        os.makedirs(train_class_dir, exist_ok=True)
        
        test_class_dir = os.path.join(test_dir, class_dir)
        os.makedirs(test_class_dir, exist_ok=True)
        
        # Loop through each image in the class directory
        for filename in os.listdir(class_dir_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                # Read the image
                image_path = os.path.join(class_dir_path, filename)
                image = cv2.imread(image_path)
                
                # Preprocess the image
                preprocessed_image = preprocess_image(image)
                
                # Decide whether to add to train or test set
                if random.uniform(0, 1) < train_test_split_ratio:  # Use the defined split ratio
                    preprocessed_image_path = os.path.join(train_class_dir, filename)
                    train_image_paths.append((preprocessed_image, preprocessed_image_path))
                else:
                    preprocessed_image_path = os.path.join(test_class_dir, filename)
                    test_image_paths.append((preprocessed_image, preprocessed_image_path))
                
                # Increment processed images counter
                processed_images += 1
                
                # Calculate progress percentage
                progress_percent = processed_images / total_images * 100
                
                # Display progress
                print(f"Processing: {processed_images}/{total_images} images ({progress_percent:.2f}%)", end='\r')

# Save train and test images
for img, path in train_image_paths:
    cv2.imwrite(path, img)

for img, path in test_image_paths:
    cv2.imwrite(path, img)

# Confirmation message
print("\nPreprocessing completed. Train and test images saved in:", processed_dir)