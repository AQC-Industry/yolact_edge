import numpy as np
import cv2 as cv
from PIL import Image

def equalize(image: np.array) -> np.array:
    """
    Histogram equalization for RGB images
    Attributes:
        image: RGB image as numpy array [np.array]
    Returns:
        img_output: Equalized RGB image as numpy array [np.array]
    """
    # Convert to YUV
    img_yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)

    # Equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv.equalizeHist(img_yuv[:,:,0])

    # Convert the YUV image back to RGB format
    img_output = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)

    return img_output

def clahe(image: np.array) -> np.array:
    """
    Histogram equalization for RGB images using CLAHE algorithm
    Attributes:
        image: RGB image as numpy array [np.array]
    Returns:
        img_output: Equalized RGB image as numpy array [np.array]
    """
    # Convert to YUV
    img_yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)

    # Equalize the histogram of the Y channel
    clh = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_yuv[:,:,0] = clh.apply(img_yuv[:,:,0])

    # Convert the YUV image back to RGB format
    img_output = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)

    return img_output

def edge_detection(image: np.array) -> np.array:
    """
    Edge detection using Canny algorithm
    Attributes:
        image: grayscale or RGB image as numpy array [np.array]
    Returns:
        img_output: normalized resultant image after applying low pass filter [np.array]
    """
    # convert to YUV
    img_yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)

    # edge detection with canny algorithm
    img_yuv[:,:,0] = cv.Canny(img_yuv[:,:,0], 100, 200)
    
    # convert back to RGB
    img_output = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)

    return img_output

def fft(image: np.array, percentile : float = 98) -> Image:
    """
    Fast Fourier Transform (FFT) an image and apply a low pass filter,
    high pass filter or band pass filter.
    The percentile values goes from 0-100, 0 means no filter and 100 complete filter.
    The recommended value for a low pass filter is percentile = 98 and
    percentile_high = 0.
    Attributes:
        image: grayscale or RGB image as numpy array [np.array]
        percentile: percentile value to apply a low pass filter [float]
    Returns:
        img_output: normalized resultant image after applying low pass filter [np.array]
    """
    # fft
    im_fft = np.fft.fftn(image)
    # Shifts the zero-frequencies to the middle
    dft_shift = np.fft.fftshift(im_fft)
    # Low pass filter
    if len(image.shape) > 2:
        magnitude_center = 20 * np.log(np.linalg.norm(dft_shift, axis=2))  # RGB
    else:
        magnitude_center = 20 * np.log(1 + np.abs(dft_shift))  # Gray
    # Mask to apply low pass filter
    mask_indx = np.where(magnitude_center > np.percentile(magnitude_center, percentile))
    mask = np.ones(image.shape, np.uint8)
    mask[mask_indx] = 0
    # Applying mask to spectral image
    f_reduced = dft_shift * mask
    # Inverse shift fft
    f_ishift = np.fft.ifftshift(f_reduced)
    # Inverse fft
    img_output = np.fft.ifftn(f_ishift).real
    # Image normalization
    img_output = np.interp(img_output, (img_output.min(), img_output.max()), (0, 255))
    img_output = img_output.astype("uint8")
    img_output = Image.fromarray(img_output)

    return img_output
