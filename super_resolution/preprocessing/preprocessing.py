import cv2

from ..params import TARGET_SIZE, CROP_SIZE

def resize_image(image, TARGET_SIZE):
    resized_image = cv2.resize(image, TARGET_SIZE)
    return resized_image

def crop_image(image, CROP_SIZE):
    height, width = image.shape[0], image.shape[1]
    y_center, x_center = int(height/2), int(width/2)
    crop_half_y, crop_half_x = int(CROP_SIZE[0]/2), int(CROP_SIZE[1]/2)
    cropped_image = image[y_center - crop_half_y:y_center + crop_half_y, x_center - crop_half_x:x_center + crop_half_x]
    return cropped_image
