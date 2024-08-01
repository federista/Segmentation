import os
from PIL import Image

def resize_images(input_dir, output_dir, size):
    """
    Resize images in the input_dir and save them to the output_dir with the same filenames.

    :param input_dir: Directory with the original images.
    :param output_dir: Directory where resized images will be saved.
    :param size: Tuple specifying the target size (width, height).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)
            img_resized = img.resize(size, Image.LANCZOS)
            output_path = os.path.join(output_dir, filename)
            img_resized.save(output_path)
            print(f"Resized and saved {filename} to {output_path}")

# Example usage
input_directory = '/home/rack_dl/segmentation/Semantic-Shapes/images'
output_directory = '/home/rack_dl/segmentation/Semantic-Shapes/images'
target_size = (640, 640)  # Change to your desired size

resize_images(input_directory, output_directory, target_size)
