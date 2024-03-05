import warnings
warnings.filterwarnings('ignore')

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img

# Define your ImageDataGenerator with the desired augmentations
data_gen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='constant'
)

def process_and_augment_images(src_dir, dest_dir, num_augmented_images=2):
    """
    Process and augment images from a source directory and save them to a destination directory.

    Args:
    - src_dir: Path to the source directory containing subdirectories of images.
    - dest_dir: Path to the destination directory where augmented images will be saved.
    - num_augmented_images: Number of augmented images to generate per original image.
    """
    for subdir, dirs, files in os.walk(src_dir):
        for filename in files:
            # Construct the full file path
            file_path = os.path.join(subdir, filename)
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG')):
                # Load the image
                img = load_img(file_path)
                img_array = img_to_array(img)
                img_array = img_array.reshape((1,) + img_array.shape)  # Reshape to (1, height, width, channels)

                # Create a subdirectory in the destination directory corresponding to the current subdirectory
                relative_path = os.path.relpath(subdir, src_dir)  # Get the relative path to maintain the folder structure
                new_subdir = os.path.join(dest_dir, relative_path)
                os.makedirs(new_subdir, exist_ok=True)
                
                # Save the original image
                original_img_save_path = os.path.join(new_subdir, f'original_{filename}')
                img.save(original_img_save_path)

                # Generate and save augmented images
                image_gen = data_gen.flow(img_array, batch_size=1, save_to_dir=new_subdir, save_prefix='aug', save_format='jpeg')
                for _ in range(num_augmented_images):
                    next(image_gen)
                    print("here")
            print("here2")

# Directories setup
base_dir = '../'  # Set your base directory path
#src_dirs = ['untouched_data/train', 'untouched_data/test', 'untouched_data/validation']
src_dirs = ['untouched_data/validation']
#dest_dirs = ['data/train', 'data/test', 'data/validation']
dest_dirs = ['data/validation']

# Process each set: train, test, validate
for src, dest in zip(src_dirs, dest_dirs):
    src_dir = os.path.join(base_dir, src)
    dest_dir = os.path.join(base_dir, dest)
    process_and_augment_images(src_dir, dest_dir)

print("Augmentation completed.")
