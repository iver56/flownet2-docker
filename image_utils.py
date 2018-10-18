import base64
import six
import tempfile

import numpy as np
from PIL import Image


def base64_png_image_to_pillow_image(base64_string):
    img_data = base64.b64decode(str(base64_string))  # Decode base64
    image = Image.open(six.BytesIO(img_data))  # Decode the PNG data
    return image


def get_apt_image_size(image, max_num_pixels):
    """
    If the image is too large, then suggest a smaller size that is below max_num_pixels.

    This is needed to avoid out of memory errors and/or excessive processing times.

    :param image:
    :param max_num_pixels:
    :return:
    """
    target_size = image.size
    while True:
        num_pixels = target_size[0] * target_size[1]
        if num_pixels <= max_num_pixels:
            break
        else:
            target_size = (
                target_size[0] * 0.98,
                target_size[1] * 0.98,
            )

    target_size = (
        int(target_size[0]),
        int(target_size[1]),
    )
    return target_size


def get_temp_png_file_path():
    return tempfile.NamedTemporaryFile(delete=False).name + '.png'
