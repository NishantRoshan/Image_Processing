import cv2
import numpy as np

def solution(image_path):
    ############################
    ############################

    # Load the image from a specified path
    image = cv2.imread(image_path)

    # Get the dimensions of the image (width, height, and channels)
    image_width, image_height, image_channels = image.shape

    # Add a white border around the image, centering it within a larger canvas
    border_thickness_horizontal = int(image_height / 2)
    border_thickness_vertical = int(image_width / 2)
    border_color = (255, 255, 255)  # White color
    image = cv2.copyMakeBorder(image, border_thickness_horizontal, border_thickness_horizontal,
                            border_thickness_vertical, border_thickness_vertical,
                            cv2.BORDER_CONSTANT, value=border_color)

    # Get the new dimensions of the image after adding the border
    new_width, new_height, _ = image.shape

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary inverse thresholding to create a binary image
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)

    # Find the coordinates of white (non-zero) pixels in the binary image
    white_pixel_coordinates = []

    for x in range(new_width):
        for y in range(new_height):
            if binary_image[x, y] > 0:
                white_pixel_coordinates.append([y, x])

    # Convert the list of coordinates to a NumPy array
    white_pixel_coordinates = np.array(white_pixel_coordinates)

    # Find the minimum area rectangle that encloses the white pixels
    min_area_rect = cv2.minAreaRect(white_pixel_coordinates)
    box_points = cv2.boxPoints(min_area_rect)
    
    box_points = box_points - box_points[0]

    # Sort the corner points of the rectangle by their distance from the top-left corner
    box_points = sorted(box_points, key=lambda p: p[0]**2 + p[1]**2)

    # Calculate the rotation angle of the rectangle in degrees
    rotation_angle = np.rad2deg(np.arctan(box_points[2][1] / box_points[2][0]))

    # Ensure the rotation angle is positive
    if rotation_angle < 0:
        rotation_angle += 180

    # Rotate the image by the calculated angle around its center
    rotation_matrix = cv2.getRotationMatrix2D((new_height / 2, new_width / 2), rotation_angle, 1.0)
    image = 255 - image
    image = cv2.warpAffine(image, rotation_matrix, (new_height, new_width), flags=cv2.INTER_LINEAR)
    image = 255 - image


    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    # image = cv2.imread(image_path)
    return image
