import numpy as np
import cv2 as cv
from PIL import Image

def crop_image(image: np.array, save_path: str, image_name: str, size: tuple = (512,512)) -> list:
    images = []
    h,w,c = image.shape
    
    unit = [int(np.floor(h/size[0])), int(np.floor(w/size[1]))]
    decimal = [h/size[0]%1, w/size[1]]
    
    # Decide when to add the missing last partition
    if decimal[0] > 0:
        step_h = unit[0]
    else:
        step_h = unit[0]
    if decimal[1] > 0:
        step_w = unit[1]
    else:
        step_w = unit[1]
    
    
    for i in range(step_w):
        for j in range(step_h):
            if j+1 < step_h and i+1 < step_w: # Selecting images inside the right and bottom borders
                images.append(image[size[0]*j:size[0]*(j+1), size[1]*i:size[1]*(i+1)])
                
            elif j+1 == step_h and i+1 < step_w and decimal[1] > 0: # Selecting images on the bottom border
                images.append(image[-size[0]:, size[1]*i:size[1]*(i+1)])
                
            elif j+1 < step_h and i+1 == step_w and decimal[0] > 0: # Selection images on the right border
                images.append(image[size[0]*j:size[0]*(j+1), -size[1]:])
                
            elif j+1 == step_h and i+1 == step_h:
                images.append(image[-size[0]:, -size[1]:])

            # Save cropped images
            im = Image.fromarray(images[-1])
            im.save(f"{save_path}/{image_name}_{i}_{j}.png")
                
    return