import cv2
import numpy as np
# from pyimage.transform

# Usage
def solution(image_path):
    image= cv2.imread(image_path)
    ######################################################################
    ######################################################################
    #####  WRITE YOUR CODE BELOW THIS LINE ###############################

    # Convert the image to grayscale and apply binary thresholding
    binary_image = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 50, 255, cv2.THRESH_BINARY)[1]

    # Find the largest contour and approximate a polygon from it
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    approximated_polygon = cv2.approxPolyDP(largest_contour, 0.04 * cv2.arcLength(largest_contour, True), True)

    # Extract and sort the vertices of the polygon
    vertices = approximated_polygon.squeeze()  # Remove extra dimension
    
    # Sort the vertices by x and y coordinates
    xs = vertices[vertices[:,0].argsort()]
    ys = vertices[vertices[:,1].argsort()]

    # Initialize corner variables
    tl, tr, bl, br = [], [], [], []

    # Check if the quadrilateral is oriented horizontally
    if np.abs(xs[0][0] - xs[1][0]) < 8 or np.abs(xs[2][0] - xs[3][0]) < 8:
        # Sort the vertices based on their x-coordinates
        sorted_vertices = sorted(vertices, key=lambda p: p[0])

        # Assign the left and right vertices
        left_vertices = sorted_vertices[:2]
        right_vertices = sorted_vertices[2:]

        # Sort the left vertices by their y-coordinates
        left_sorted = sorted(left_vertices, key=lambda p: p[1])
        tl, bl = left_sorted

        # Sort the right vertices by their y-coordinates
        right_sorted = sorted(right_vertices, key=lambda p: p[1])
        tr, br = right_sorted

    # Check if the quadrilateral is oriented vertically
    elif np.abs(ys[0][1] - ys[1][1]) < 8 or np.abs(ys[2][1] - ys[3][1]) < 8:
        # Sort the topmost vertices by x-coordinate
        top_sorted = ys[:2][ys[:2,0].argsort()]
        # print(top_sorted)
        tl, tr = top_sorted[0], top_sorted[1]

        # Sort the bottommost vertices by x-coordinate
        bottom_sorted = ys[2:][ys[2:,0].argsort()]
        bl, br = bottom_sorted[0], bottom_sorted[1]

    # Swap row and column coordinates to match image indexing
    tl, tr, bl, br = [(p[1], p[0]) for p in [tl, tr, bl, br]]

    # Now, you have tl, tr, bl, and br representing the corners of the quadrilateral.
    
    
    # Create a white canvas
    canvas = np.full((600, 600, 3), 255, dtype="uint8")

    # Check if the top-left and top-right corners have the same color
    if np.array_equal(image[tl], image[tr]):
        # Draw rectangles in the top and bottom halves with the same color
        b, g, r = image[tl]
        cv2.rectangle(canvas, (0, 0), (600, 199), (int(b), int(g), int(r)), -1)
        b, g, r = image[bl]
        cv2.rectangle(canvas, (0, 399), (600, 600), (int(b), int(g), int(r)), -1)
    else:
        # Draw rectangles in the left and right halves with respective colors
        b, g, r = image[tl]
        cv2.rectangle(canvas, (0, 0), (199, 600), (int(b), int(g), int(r)), -1)
        b, g, r = image[tr]
        cv2.rectangle(canvas, (399, 0), (600, 600), (int(b), int(g), int(r)), -1)
        

    # Define circle parameters
    center = (299, 299)
    radius = 99
    color = (125, 0, 0)  # Dark blue color in BGR format
    thickness_circle = 2
    thickness_lines = 1

    # Draw a circle
    cv2.circle(canvas, center, radius, color, thickness_circle)

    # Draw lines within the circle
    for i in range(24):
        angle = np.deg2rad(i * 15)
        x = int(center[0] + radius * np.cos(angle))
        y = int(center[1] + radius * np.sin(angle))
        cv2.line(canvas, center, (x, y), color, thickness_lines)

    # Update the image with the canvas
    image = canvas
    


    ######################################################################

    return image
