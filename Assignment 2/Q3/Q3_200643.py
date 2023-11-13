import cv2
import numpy as np

def calculate_distance(point1, point2):
    return max(abs(point1[0] - point2[0]), abs(point1[1] - point2[1]))

def preprocess_image(input_image):
    input_image = cv2.copyMakeBorder(input_image, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    input_image = cv2.bilateralFilter(input_image, 15, 75, 75)
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    inverted_image = 255 - gray_image
    _, thresholded_image = cv2.threshold(inverted_image, 25, 255, cv2.THRESH_TOZERO)

    corners = cv2.goodFeaturesToTrack(thresholded_image, 500, 0.01, 5)
    corners = np.intp(corners).squeeze()

    return thresholded_image, corners

def identify_top_bottom_points(corners, thresholded_image):
    top_points = []
    bottom_points = []

    for x, y in corners:
        if thresholded_image[y, x - 5] == 0 and thresholded_image[y, x + 5] == 0 and thresholded_image[y - 5, x] == 0:
            top_points.append((x, y))
        if thresholded_image[y + 5, x] == 0:
            bottom_points.append((x, y))

    return top_points, bottom_points

def classify_image(top_points, corners):
    image_class = 'real'

    max_y_coordinate = max(corners, key=lambda x: x[1])[1]
    mid_points = []

    for x, y in corners:
        if max_y_coordinate - y <= 5:
            mid_points.append(x)

    mid_point = int((min(mid_points) + max(mid_points)) / 2)

    left_point_count = 0
    right_point_count = 0

    for x, y in top_points:
        if mid_point - x > 10:
            left_point_count += 1
        if x - mid_point > 10:
            right_point_count += 1

    if left_point_count != 4 or right_point_count != 5:
        image_class = 'fake'

    mid_line = [(mid_point, int((min(corners, key=lambda x: x[1])[1] + max_y_coordinate) / 2))]
    reference_point = mid_line[0]

    distance_threshold = (mid_point - max([x for x, y in top_points if mid_point - x > 10])) / 2
    within_threshold_count = 0

    for x, y in corners:
        if calculate_distance(reference_point, (x, y)) < distance_threshold:
            within_threshold_count += 1

    if within_threshold_count < 5:
        image_class = 'fake'

    return image_class

def solution(audio_path):
    input_image = cv2.imread(audio_path)
    thresholded_image, corners = preprocess_image(input_image)
    top_points, bottom_points = identify_top_bottom_points(corners, thresholded_image)

    image_class = classify_image(top_points, corners)

    return image_class

# Example usage:
# result = solution('path_to_image.jpg')
# print(result)
