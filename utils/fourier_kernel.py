import torch
from pyefd import elliptic_fourier_descriptors
import numpy as np
import cv2
import albumentations as A


def fourier_kernel(img, order=14, points=40000, fill=False):
    img = img
    contours = get_contour(img)
    coeffs = get_coeffs(contours=contours, order=order, img=img)
    x, y = get_xy(coeffs=coeffs, img=img, points=points)
    kernel = get_kernel(x, y)
    if fill:
        kernel = fill_contours(kernel)*1
    return kernel

def resize_kernel(kernel, k_size=19):
    ab = A.Compose([
        A.Resize(k_size, k_size)
    ])
    mask_kernel = kernel.astype(np.uint8)
    transformed = ab(image=mask_kernel, mask=mask_kernel)
    res = transformed["mask"]
    return res

def get_contour(img):
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,0)
    contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_coeffs(contours, order, img):   
    contours = np.vstack(contours).squeeze()
    coeffs = elliptic_fourier_descriptors(contours, order=order)
    return coeffs
    
def get_xy(coeffs, img, points=None):
    locus = (int(img.shape[0]/2),int(img.shape[0]/2))
    if points is None:
        points = max(img.shape[0]*2, 500)
    m=points
    xt = np.ones((m,)) * locus[0]
    yt = np.ones((m,)) * locus[1]
    t = np.linspace(0, 1.0, m)
    
    for n in range(coeffs.shape[0]):
        xt += (coeffs[n, 0] * np.cos(2 * (n + 1) * np.pi * t)) + (
            coeffs[n, 1] * np.sin(2 * (n + 1) * np.pi * t)
        )
        yt += (coeffs[n, 2] * np.cos(2 * (n + 1) * np.pi * t)) + (
            coeffs[n, 3] * np.sin(2 * (n + 1) * np.pi * t)
        )
    return xt, yt

def get_kernel(xt, yt):
    m = min(xt.shape[0], yt.shape[0])
    kernel_mean = np.zeros((m, m))
    for k in range(m):
        i = int(yt[k])
        j = int(xt[k])
        kernel_mean[i][j] = 1.
    kernel_mean = kernel_mean[int(yt.min())-2: int(yt.max())+2, 
                              int(xt.min())-2: int(xt.max())+2]
    return kernel_mean

def fill_contours(arr):
    return np.logical_and(np.maximum.accumulate(arr,1), 
                          np.maximum.accumulate(arr[:,::-1],1)[:,::-1])