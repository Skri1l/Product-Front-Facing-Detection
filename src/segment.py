import cv2


def segment_products(enhanced):
    """
    Create a binary mask of visually relevant product/label regions.
    It uses HSV color segmentation + edge-based segmentation.
    """
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)

    # Colorful labels/products
    red1 = cv2.inRange(hsv, (0, 60, 50), (12, 255, 255))
    red2 = cv2.inRange(hsv, (168, 60, 50), (180, 255, 255))
    blue = cv2.inRange(hsv, (85, 50, 50), (135, 255, 255))
    yellow = cv2.inRange(hsv, (12, 50, 60), (38, 255, 255))
    green = cv2.inRange(hsv, (38, 40, 50), (90, 255, 255))
    orange = cv2.inRange(hsv, (5, 50, 60), (25, 255, 255))

    # Light labels / white bottles, but not too permissive
    white = cv2.inRange(hsv, (0, 0, 160), (180, 55, 255))

    color_mask = red1 | red2 | blue | yellow | green | orange | white

    # Edge mask helps with labels and product contours
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 160)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, edge_kernel, iterations=1)

    mask = cv2.bitwise_or(color_mask, edges)
    return mask
