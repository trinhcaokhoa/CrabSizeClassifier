import os
import albumentations as A
import cv2
import random

# Declare an augmentation pipeline
transform = A.Compose([
    A.Rotate(limit=40),
    A.RandomBrightnessContrast(p=random.randrange(0,1)),
    A.ImageCompression(quality_lower=85, quality_upper=100, p=random.randrange(0,1)),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=random.randrange(0,1)),
    A.HorizontalFlip(p=random.randrange(0, 1)),
])

# declare image folder
path = "crab_raw_data/data/train/dungeness_crab_small"


def main():
    # Read an image with OpenCV and convert it to the RGB colorspace
    for image in os.listdir(path):
        image_dir = os.path.join(path, image)
        # check the valid image is passed and read image
        if image.endswith(".jpg"):
            crab_image = cv2.imread(image_dir, 1)
            crab_image = cv2.cvtColor(crab_image, cv2.COLOR_BGR2RGB)

            # Augment an image

            transformed = transform(image=crab_image)
            transformed_image = transformed["image"]

            # Rename the image to avoid overwriting
            os.rename(os.path.join(path, image), os.path.join(path, str(random.randint(0,1000)) + image))
            image_dir = os.path.join(path, image)
            cv2.imwrite(image_dir, transformed_image)
        else:
            continue
    return print("Image albumentations")


main()
