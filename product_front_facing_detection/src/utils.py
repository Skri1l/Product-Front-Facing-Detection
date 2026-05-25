import os
import cv2


def create_directory(path):
    """
    Creates directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def read_image(image_path):
    """
    Reads image from disk.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Cannot open image: {image_path}"
        )

    return image


def save_image(path, image):
    """
    Saves image to disk.
    """

    cv2.imwrite(str(path), image)