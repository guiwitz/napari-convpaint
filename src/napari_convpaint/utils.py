import skimage
import numpy as np

def generate_synthetic_square(im_dims, square_dims, rgb=True):

    square = 10*np.ones(square_dims, dtype=np.uint8)
    if rgb is True:
        im = np.random.randint(0, 5, (im_dims[0], im_dims[1], 3), dtype=np.uint8)
        im[im_dims[0]//2 - square_dims[0]//2:im_dims[0]//2 + square_dims[0]//2,
        im_dims[1]//2 - square_dims[1]//2:im_dims[1]//2 + square_dims[1]//2, 0] += square
    else:
        im = np.random.randint(0, 5, (im_dims[0], im_dims[1]), dtype=np.uint8)
        im[im_dims[0]//2 - square_dims[0]//2:im_dims[0]//2 + square_dims[0]//2,
        im_dims[1]//2 - square_dims[1]//2:im_dims[1]//2 + square_dims[1]//2] += square
    
    ground_truth = np.zeros(im_dims, dtype=np.uint8)
    ground_truth[im_dims[0]//2 - square_dims[0]//2:im_dims[0]//2 + square_dims[0]//2,
                    im_dims[1]//2 - square_dims[1]//2:im_dims[1]//2 + square_dims[1]//2] = 1

    return im, ground_truth

def generate_synthetic_circle_annotation(im_dims, circle1_xy, circle2_xy):

    circle1 = skimage.draw.disk(circle1_xy, 5)
    circle2 = skimage.draw.disk(circle2_xy, 5)
    annotation = np.zeros(im_dims, dtype=np.uint8)
    annotation[circle1] = 1
    annotation[circle2] = 2

    return annotation