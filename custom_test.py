import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import numpy as np
import tensorflow as tf
import segmentation_models as sm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Define paths
test_image_dir = '/home/rack_dl/segmentation/Semantic-Shapes/images'  # Directory containing test images
mask_dir = '/home/rack_dl/segmentation/Semantic-Shapes/masks'  # Directory containing ground truth masks

# Define the target size of the images
target_size = (256, 256)

num_classes = 3  # Update this based on your number of classes

# Define the color map for your classes
'''
color_map = np.array([
    [255, 0, 0],                  
    [0, 255, 0],       
    [255,255,0],    
    [0,255,255],     
    [0, 0, 255],
    [0, 0, 0]
])
'''

color_map = np.array([
    [255, 0, 0],                  
    [0, 255, 0],
    [0, 0, 0]       
])

# Load the trained model
model = tf.keras.models.load_model('/home/rack_dl/segmentation/Semantic-Shapes/models/segmentation_model.h5', custom_objects={'Unet': sm.Unet})

def load_and_preprocess_image(image_file, target_size=target_size):
    """Load and preprocess the image for prediction."""
    img = load_img(image_file, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_image(image_file, model, target_size=target_size):
    """Predict the segmentation mask for a single image."""
    img_array = load_and_preprocess_image(image_file, target_size)
    pred_mask = model.predict(img_array)
    pred_mask = np.squeeze(pred_mask)  # Remove batch dimension
    pred_mask = np.argmax(pred_mask, axis=-1)  # Convert to class labels
    return pred_mask

def apply_color_map(pred_mask):
    """Apply the color map to the predicted mask."""
    colored_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    for class_index, color in enumerate(color_map):
        colored_mask[pred_mask == class_index] = color
    return colored_mask

def load_mask(mask_file, target_size=target_size):
    """Load and preprocess the ground truth mask."""
    mask = load_img(mask_file, target_size=target_size)
    mask = img_to_array(mask).astype(np.uint8)
    class_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    for class_index, color in enumerate(color_map):
        class_mask[np.all(mask == color, axis=-1)] = class_index
    #mask = np.argmax(mask, axis=-1)  # Convert to class labels
    return class_mask

def visualize_predictions(test_image_dir, mask_dir, model):
    """Visualize predictions with Matplotlib."""
    image_files = [os.path.join(test_image_dir, f) for f in os.listdir(test_image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    mask_files = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')]
    current_index = 0

    def show_image(image_file, mask_file):
        img = load_img(image_file)#, color_mode= 'grayscale' , target_size=target_size)
        img_array = img_to_array(img) / 255.0
        
        pred_mask = predict_image(image_file, model)
        colored_pred_mask = apply_color_map(pred_mask)
        
        print("Mask_path: ", mask_file)
        ground_truth_mask = load_mask(mask_file)
        colored_ground_truth_mask = apply_color_map(ground_truth_mask)
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        
        axes[0].imshow(img_array)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        #axes[1].imshow(colored_ground_truth_mask)
        #axes[1].set_title('Ground Truth')
        #axes[1].axis('off')
        
        axes[1].imshow(colored_pred_mask)
        axes[1].set_title('Predicted Mask')
        axes[1].axis('off')
        
        plt.show()
    
    show_image(image_files[current_index], mask_files[current_index])
    
    while True:
        user_input = input("Press 'n' to view the next image, 'p' to view the previous image, or 'q' to quit: ").strip().lower()
        if user_input == 'n':
            current_index = (current_index + 1) % len(image_files)
            plt.close()
            show_image(image_files[current_index], mask_files[current_index])
        elif user_input == 'p':
            current_index = (current_index - 1) % len(image_files)
            show_image(image_files[current_index], mask_files[current_index])
        elif user_input == 'q':
            break

# Run the visualization
visualize_predictions(test_image_dir, mask_dir, model)