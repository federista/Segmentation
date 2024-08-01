import cv2
import os

# Function to apply NLM filter to an image and save the result
def apply_nlm_filter(image_path, output_path, h=10, templateWindowSize=7, searchWindowSize=21):
    # Read the image
    sar_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply the Non-Local Means (NLM) filter
    filtered_image = cv2.fastNlMeansDenoising(sar_image, None, h=h, templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize)
    
    # Save the filtered image
    cv2.imwrite(output_path, filtered_image)

# Define the input directory containing the images
input_dir = '/home/rack_dl/segmentation/Semantic-Shapes/images'
output_dir = '/home/rack_dl/segmentation/Semantic-Shapes/images'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process all images in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        apply_nlm_filter(input_path, output_path)

# Optionally, display the last processed images
#cv2.imshow('Original SAR Image', sar_image)
#cv2.imshow('Filtered SAR Image', filtered_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
