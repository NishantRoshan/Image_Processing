import cv2
import numpy as np

def solution(image_path_a, image_path_b):
    spatial_sigma_value = 10
    range_sigma_value = 1
    window_size = 11

    # Load images
    no_flash_image = cv2.imread(image_path_a)
    flash_image = cv2.imread(image_path_b)

    # Create Gaussian kernel
    matrix = np.exp(-np.arange(256) * np.arange(256) / (2 * (range_sigma_value**2)))
    xx = -window_size + np.arange(2 * window_size + 1)
    yy = -window_size + np.arange(2 * window_size + 1)
    x, y = np.meshgrid(xx, yy)
    spatial_gaussian = np.exp(-(x ** 2 + y ** 2) / (2 * (spatial_sigma_value**2)))

    # Pad images with mirror reflections
    no_flash_image_padded = np.pad(no_flash_image, ((window_size, window_size), (window_size, window_size), (0, 0)), mode='reflect')
    flash_image_padded = np.pad(flash_image, ((window_size, window_size), (window_size, window_size), (0, 0)), mode='reflect')

    bilateral_result = np.zeros_like(flash_image)

    # Apply bilateral filter
    for i in range(flash_image.shape[0]):
        for j in range(flash_image.shape[1]):
            for k in range(flash_image.shape[2]):
                i_padded = i + window_size
                j_padded = j + window_size

                neighbourhood = flash_image_padded[i_padded - window_size: i_padded + window_size + 1, j_padded - window_size: j_padded + window_size + 1, k]
                central = flash_image_padded[i_padded, j_padded, k]

                range_gaussian = matrix[abs(neighbourhood - central)]
                space_gaussian = spatial_gaussian

                result = range_gaussian * space_gaussian
                normalization = np.sum(result)

                result = result / normalization
                result = result * no_flash_image_padded[i_padded - window_size: i_padded + window_size + 1, j_padded - window_size: j_padded + window_size + 1, k]

                bilateral_result[i, j, k] = np.sum(result)

    return bilateral_result
