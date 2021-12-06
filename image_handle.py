import cv2


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def unsharp(img):
    gaussian_3 = cv2.GaussianBlur(img, (0, 0), 2.0)
    return cv2.addWeighted(img, 2.0, gaussian_3, -1.0, 0)
