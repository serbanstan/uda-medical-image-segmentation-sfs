import numpy as np

from PIL import Image
import cv2

def rotate(X, Y, angle=90):
    # Rotates and image/label pair by the given angle

    assert X.shape == Y.shape
    assert len(X.shape) == 3 and X.shape[-1] == 3

    rotX = np.zeros(X.shape)
    rotY = np.zeros(Y.shape, dtype=np.int32)

    for c in range(3):
        curr_slice = Image.fromarray(X[..., c])
        rotX[...,c] = np.array(curr_slice.rotate(angle))

        curr_slice = Image.fromarray(Y[..., c])
        rotY[...,c] = np.array(curr_slice.rotate(angle))

    assert rotX.shape == X.shape and rotY.shape == Y.shape
    assert 0 <= np.min(rotY) and np.max(rotY) <= 4

    return (rotX, rotY)

def zoom(X, Y, border_change=50):
    # Performs a random zoom-in augmentation
    # The borders of the zoomed-in region will have a at most border_change displacement from the original frame

    newX = np.copy(X)
    newY = np.copy(Y)

    x1 = np.random.randint(50)
    y1 = np.random.randint(50)
    x2 = newX.shape[0]-np.random.randint(50)
    y2 = newX.shape[1]-np.random.randint(50)
    newX = np.copy(newX[x1:x2, y1:y2, :])
    newY = np.copy(newY[x1:x2, y1:y2, :])
    
    newX = cv2.resize(newX, (256,256), interpolation=cv2.INTER_CUBIC)
    newY = cv2.resize(newY, (256,256), interpolation=cv2.INTER_NEAREST)

    assert 0 <= np.min(newY) and np.max(newY) <= 4

    return (newX, newY)

def add_noise(X,Y,alpha=1):
    newX = X + alpha * np.random.randn(256,256,3)
    return (newX, np.copy(Y))

def invert(X,Y):
    return np.copy(-X),np.copy(Y)

def combined(X, Y):
    # Creates an augmented image combining all of the above

    newX = np.copy(X)
    newY = np.copy(Y)

    if np.random.randint(2) == 1:
        newX,newY = add_noise(newX,newY, np.random.randn())

    if np.random.randint(2) == 1:
        newX,newY = invert(newX,newY)

    if np.random.randint(5) < 4:
        newX,newY = rotate(newX, newY, np.random.randint(40) - 20)

    if np.random.randint(5) < 4:
        newX,newY = zoom(newX, newY, 70)

    return (newX,newY)
