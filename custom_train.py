import os
os.environ["SM_FRAMEWORK"] = "tf.keras"


import os
import numpy as np
import tensorflow as tf
import segmentation_models as sm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import json
import cv2

# Configuration
image_size = (256, 256) # Do not change this (Resizing resolution for model)
num_classes = 3 # 5 classes + 1 background
json_to_mask_size = (256,256) #size of image
# Label for conversion of JSON to Mask
'''
label_map = {
    'runway': [255, 0, 0],   
    'grass': [0, 255, 0],     
    'chinook': [255,255,0],
    'building': [0,255,255],
    'hanger': [0,0,255], 
    'background': [0, 0, 0]   # Black
}
'''
label_map = {
    'ship': [255, 0, 0],   
    'shore': [0, 255, 0],      
    'background': [0, 0, 0]   # Black
}
def generate_missing_json():
    for im in os.listdir('images'):
        fn = im.split('.')[0]+'.json'
        path = os.path.join('annotated', fn)
        if not os.path.exists(path):
            json_dict = {
                'shapes': [{"label": "background",
                            "points": [[0,0],
                                       [0, image_size[0]-1],
                                       [image_size[0]-1, image_size[1]-1],
                                       [image_size[0]-1, 0]]}]
            }
            with open(path, 'w') as handle:
                json.dump(json_dict, handle, indent=2)

# Check the length of annotation files
if len(os.listdir('images')) != len(os.listdir('annotated')):
    generate_missing_json()

def load_image(image_file):
    img = load_img(image_file, target_size=image_size)
    img = img_to_array(img) / 255.0
    return img

def rgb_to_class_index(mask, label_map):
    class_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for label, color in label_map.items():
        color = np.array(color, dtype=np.uint8)
        #print(f"{color} : {label} ")
        class_mask[np.all(mask == color, axis=-1)] = list(label_map.keys()).index(label)
    return class_mask

def load_mask(mask_file):
    mask = load_img(mask_file, target_size=image_size)
    mask = img_to_array(mask).astype(np.uint8)
    #print(f"Mask:  {np.unique(mask)}")
    mask = rgb_to_class_index(mask, label_map)
    return mask

def prepare_dataset(image_dir, mask_dir):
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]
    mask_files = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')]

    images = np.array([load_image(f) for f in image_files])
    masks = np.array([load_mask(f) for f in mask_files])
    masks = to_categorical(masks, num_classes=num_classes)  # One-hot encode the masks

    return tf.data.Dataset.from_tensor_slices((images, masks))

def json_to_mask(json_file, output_file, label_map, img_shape=json_to_mask_size):
    with open(json_file) as f:
        data = json.load(f)
    
    img = np.zeros((*img_shape, 3), dtype=np.uint8)
    for shape in data['shapes']:
        label = shape['label']
        points = np.array(shape['points'], dtype=np.int32)
        if label in label_map:
            color = label_map[label]
            cv2.fillPoly(img, [points], color=color)
    
    cv2.imwrite(output_file, img)

def convert_all_json_to_masks(json_dir, mask_dir, label_map):
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    for file_name in os.listdir(json_dir):
        if file_name.endswith('.json'):
            json_file = os.path.join(json_dir, file_name)
            mask_file = os.path.join(mask_dir, file_name.replace('.json', '.png'))
            json_to_mask(json_file, mask_file, label_map)

# Convert JSON to masks
image_dir = '/home/rack_dl/segmentation/Semantic-Shapes/images'
json_dir = '/home/rack_dl/segmentation/Semantic-Shapes/annotated'
mask_dir = '/home/rack_dl/segmentation/Semantic-Shapes/masks'

convert_all_json_to_masks(json_dir, mask_dir, label_map)

# Load dataset
dataset = prepare_dataset(image_dir, mask_dir)
dataset = dataset.shuffle(buffer_size=100).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)

# Define model
backbone = 'resnet34'
model = sm.Unet(backbone_name=backbone, input_shape=(None, None, 3), classes=num_classes, activation='softmax')

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define the checkpoint callback
checkpoint_path = os.path.join('models', 'segmentation_model.h5')
checkpoint_callback = ModelCheckpoint(
    checkpoint_path,
    save_best_only=True,
    save_weights_only=False,
    monitor='accuracy',
    mode='max',
    save_freq='epoch',
    period=10
)

# Load the model from the checkpoint if it exists
if os.path.exists(checkpoint_path):
    model.load_weights(checkpoint_path)
    print("Loading previous weights")

# Train model
history = model.fit(dataset, epochs=500, callbacks=[checkpoint_callback])

# Save model
# model.save('/home/rack_dl/segmentation/Semantic-Shapes/models/segmentation_model.h5')
