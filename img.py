import cv2
def preprocess(img):
    # resizing the 96x96 img to 84x84
    img = img[:84, 6:90] # only take the first 84 rows (cut off black bar at bottom) and take the columns at the center of the track
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img
