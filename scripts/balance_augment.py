import os
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img

# Configuration
input_base = "../dataset"
output_base = "../augmented_dataset"
target_count = 900  # desired images per class
img_size = (96, 96)

# Augmentation config
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

os.makedirs(output_base, exist_ok=True)

for class_name in os.listdir(input_base):
    class_input_dir = os.path.join(input_base, class_name)
    class_output_dir = os.path.join(output_base, class_name)
    os.makedirs(class_output_dir, exist_ok=True)

    original_images = os.listdir(class_input_dir)
    count = 0

    print(f"[INFO] Processing class: {class_name} | Starting with {len(original_images)} images")

    # Copy original images first
    for img_name in original_images:
        src = os.path.join(class_input_dir, img_name)
        dst = os.path.join(class_output_dir, f"orig_{img_name}")
        shutil.copy(src, dst)
        count += 1

    # Start augmentation
    while count < target_count:
        img_name = np.random.choice(original_images)
        img_path = os.path.join(class_input_dir, img_name)
        img = load_img(img_path, target_size=img_size)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        prefix = f"aug_{count}_"
        for batch in datagen.flow(x, batch_size=1):
            aug_img = array_to_img(batch[0])
            aug_img.save(os.path.join(class_output_dir, prefix + img_name))
            count += 1
            break  # Save only one image per loop

    print(f"[INFO] Finished {class_name} → {count} images\n")
