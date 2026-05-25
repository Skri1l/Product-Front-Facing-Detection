import os

from src.segment import segment_products
from src.clean import cleanProducts
from src.utils import (
    create_directory,
    read_image,
    save_image
)


INPUT_DIR = "input_images"

OUTPUT_DIR = "outputs"
SEGMENTATION_DIR = os.path.join(
    OUTPUT_DIR,
    "segmentation"
)

CLEAR_DIR = os.path.join(
    OUTPUT_DIR,
    "clear"
)


def process_image(image_name):
    input_path = os.path.join(
        INPUT_DIR,
        image_name
    )

    image = read_image(input_path)

    # Enhance Start
    # Ivan adds picture enhancement here
    # Enhance End

    # Segmentation Start
    segmentation_mask = segment_products(image)

    image_name_without_extension = os.path.splitext(
        image_name
    )[0]

    output_path = os.path.join(
        SEGMENTATION_DIR,
        f"{image_name_without_extension}_mask.png"
    )

    save_image(output_path, segmentation_mask)

    print(f"Segmented: {image_name}")
    # Segmentation End

    # Clean Start
    cleanedImage = cleanProducts(output_path)

    cleanedImageOutput = os.path.join(
        CLEAR_DIR,
        f"{image_name_without_extension}_cleaned.png"
    )

    save_image(cleanedImageOutput, cleanedImage)

    print(f"Cleaned: {image_name}")
    # Clean End


def main():
    create_directory(SEGMENTATION_DIR)
    create_directory(CLEAR_DIR)

    image_files = [
        file for file in os.listdir(INPUT_DIR)
        if file.lower().endswith(
            (".png", ".jpg", ".jpeg")
        )
    ]

    if not image_files:
        print("No input images found.")
        return

    for image_name in image_files:
        process_image(image_name)

    print("Segmentation completed.")


if __name__ == "__main__":
    main()
