import io
import cv2
import sys
sys.path.append('/home/e_radionova/cig5_Research_optimal_kernels/Dataset_parametrezation')
from pyefd import elliptic_fourier_descriptors
from pyEFD import plot_efd
import numpy as np
from matplotlib import pyplot as plt


def get_img_from_fig(fig, dpi):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.rot90(img, 1)

    return img


def check_format_of_names(checking_arr, format_name):
    error_lst = []
    error_idxs = []
    for idx, name in enumerate(checking_arr):
        if format_name not in name:
            error_lst.append(name)
            error_idxs.append(idx) 

    checking_arr = np.delete(checking_arr, error_idxs)
    return checking_arr


# methods available: ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#                    'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

def template_matching(image, template, method=cv2.TM_CCORR):
#     template = template.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(image, template, method)
    return res


def thresholding(image, thresh_coeff = None, otsu=False):
    max_value = np.max(image)
    if thresh_coeff == None:
        thresh_coeff = max_value / 3
    thresh = thresh_coeff * np.mean(np.unique(image))
    
    print('avg:', np.mean(np.unique(image)))
    print('max:', max_value) 
    plt.imshow(image)
    plt.title('Before thresholding')
    plt.show()
    
    if otsu: 
        image = image.astype(np.uint8)
        otsu_threshold, image_result = cv2.threshold(image, thresh, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        otsu_threshold, image_result = cv2.threshold(image, thresh, max_value, cv2.THRESH_BINARY)
    return image_result

def get_Fourier_coeffs_and_kernel(image, order, kernel_size):
    imgray = image.astype(np.uint8)
#     imgray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
#     imgray = image
#     img_max = imgray.max()
#     img_mean = imgray.mean()
#     ret,thresh = cv2.threshold(imgray, img_mean, img_max, 0)
    contours, hierarchy = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_new = np.vstack(contours).squeeze()
    coeffs = elliptic_fourier_descriptors(contours_new, order=order)
    
    locus = (0.0, 0.0)
    m = order
    xt = np.ones((m,)) * locus[0]
    yt = np.ones((m,)) * locus[1]

    t = np.linspace(0, 1.0, m)

    for n in range(coeffs.shape[0]):
        yt += (coeffs[n, 0] * np.cos(2 * (n + 1) * np.pi * t)) + (
            coeffs[n, 1] * np.sin(2 * (n + 1) * np.pi * t)
        )
        xt += (coeffs[n, 2] * np.cos(2 * (n + 1) * np.pi * t)) + (
            coeffs[n, 3] * np.sin(2 * (n + 1) * np.pi * t)
        )
    a = image.shape[0]
    b = image.shape[1]
    fig = plt.figure(figsize=(a/b, b/b))
    ax = fig.add_subplot(111)
    ax.plot(xt, yt, 'black', linewidth=3)
    ax.axis('off')
    plt.close(fig)

    kernel = get_img_from_fig(fig, dpi=kernel_size)  
    kernel = kernel / kernel.max()
    kernel = abs(kernel - 1)
#     kernel = (kernel - np.mean(kernel)) / np.std(kernel)

    return coeffs, kernel
