import os
import sys
import numpy as np
import scipy.stats as ss
import pandas as pd
import skimage.io
import skimage.color
import skimage.transform
import random
from distutils.version import LooseVersion

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config


class CocoConfig(Config):
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # Background + catagory

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 50

    VALIDATION_STEPS = 20

    # Adjust learning rate if needed
    LEARNING_RATE = 0.0001

    # Skip detections with < 70% confidence
    DETECTION_MIN_CONFIDENCE = 0.7

    DEFAULT_LOGS_DIR = './logs'


def softmax(a):
    ex_a = np.exp(a)
    # print('Sum of np.exp(a) or ex_a:', sum(ex_a))
    return ex_a / sum(ex_a)


def rel_entr(a, b):
    array_a = np.array(a, dtype=np.float64)
    array_b = np.array(b, dtype=np.float64)
    # print(array_a)
    # print(array_b)
    v = array_a / array_b
    lg = np.log(v)
    return array_a * lg


def entropy(pk, qk=None, base=None, axis=0):
    pk = np.array(pk, dtype=np.float64)
    # print('Softmax of pk')
    pk = softmax(pk)
    if qk is None:
        vec = ss.entr(pk)
    else:
        qk = np.array(qk, dtype=np.float64)
        if qk.shape != pk.shape:
            raise ValueError("qk and pk must have same shape.")
        # print('Softmax of qk')
        qk = softmax(qk)
        vec = rel_entr(pk, qk)
    S = np.sum(vec, axis=axis)
    if base is not None:
        S /= np.log(base)
    return S


def KL_div(array1, array2):
    return entropy(array1, array2)


def resize_layer(layer_w):
    length = 1
    shape = np.shape(layer_w)
    for i in range(len(shape)):
        length = length * shape[i]
    # print('length = ',length)
    flat_array = np.reshape(layer_w, length)
    # print(flat_array,dtype=float)
    return flat_array


def relative_information(dataset):
    s = resize_layer(sum(dataset) / len(dataset))
    ri = 0
    for si in dataset:
        s_flat = resize_layer(si)
        ri = ri + KL_div(s_flat, s) + KL_div(s, s_flat)
    return ri


def save_data(data, path):
    data_frame = pd.DataFrame(data)
    data_frame.to_csv(path, index=False, header=False)


def load_image(path):
    """Load the specified image and return a [H,W,3] Numpy array.
    """
    # Load image
    image = skimage.io.imread(path)
    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)
    # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image


def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    """A wrapper for Scikit-Image resize().

    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)


def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = resize(image, (round(h * scale), round(w * scale)),
                       preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop


def mold_image(images, config):
    """Expects an RGB image (or array of images) and subtracts
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.
    original_image_shape: [H, W, C] before resizing or padding.
    image_shape: [H, W, C] after resizing and padding
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    scale: The scaling factor applied to the original image (float32)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +                  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +           # size=3
        list(window) +                # size=4 (y1, x1, y2, x2) in image cooredinates
        [scale] +                     # size=1
        list(active_class_ids)        # size=num_classes
    )
    return meta


def mold_inputs(config, image, images=[]):
    """Takes a list of images and modifies them to the format expected
    as an input to the neural network.
    images: List of image matrices [height,width,depth]. Images can have
        different sizes.

    Returns 3 Numpy matrices:
    molded_images: [N, h, w, 3]. Images resized and normalized.
    image_metas: [N, length of meta data]. Details about each image.
    windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
        original image (padding excluded).
    """
    # molded_images = []
    # image_metas = []
    # windows = []
    # for image in images:
    #     # Resize image
    #     molded_image, window, scale, padding, crop = resize_image(
    #         image,
    #         min_dim=config.IMAGE_MIN_DIM,
    #         min_scale=config.IMAGE_MIN_SCALE,
    #         max_dim=config.IMAGE_MAX_DIM,
    #         mode=config.IMAGE_RESIZE_MODE)
    #     molded_image = mold_image(molded_image, config)
    #     # Build image_meta
    #     image_meta = compose_image_meta(
    #         0, image.shape, molded_image.shape, window, scale,
    #         np.zeros([config.NUM_CLASSES], dtype=np.int32))
    #     # Append
    #     molded_images.append(molded_image)
    #     windows.append(window)
    #     image_metas.append(image_meta)
    # # Pack into arrays
    # molded_images = np.stack(molded_images)
    # image_metas = np.stack(image_metas)
    # windows = np.stack(windows)
    # return molded_images, image_metas, windows

    # Resize image
    molded_image, window, scale, padding, crop = resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    molded_image = mold_image(molded_image, config)
    return molded_image


if __name__ == '__main__':
    from find_catID import coco_categories
    from PythonAPI.pycocotools.coco import COCO

    label_size = 500
    labels = coco_categories[:10]
    netL_config = CocoConfig()

    # calculate RI for each coco label (first 10 coco labels) using each label's 500 training images
    coco = COCO("../drive/My Drive/coco_datasets/annotations/instances_train2014.json")
    # coco = COCO("./cocoDS/annotations/instances_train2014.json")

    class_ids = coco.getCatIds(catNms=labels)
    # image_ids = []
    RIs = []
    for class_id in class_ids:
        # get current label
        category = (coco.loadCats(class_id))[0]
        label_name = category['name']
        print(label_name, ':')

        class_img_ids = list(coco.getImgIds(catIds=[class_id]))
        class_img_ids = list(set(class_img_ids))
        class_img_ids = class_img_ids[:label_size]
        # print('# of images:', len((class_img_ids)))

        # # TODO: get the images & also resize them as how it was prep for training
        # # Note: should the data passed into ri be the images data or mask data of the label in the images?
        class_images = []
        print('Loading and resizing images...')
        for i in class_img_ids:
            img_path = coco.imgs[i]['coco_url']
            img = load_image(img_path)
            class_images.append(mold_inputs(netL_config, img))
        class_images = np.stack(class_images)
        print('Images are loaded and resized, calculating RI...')
        class_ri = relative_information(class_images)
        print(label_name, 'RI:', class_ri)
        RIs.append({label_name: class_ri})
        print()

        # # get all labels' training data
        # if len(class_img_ids) < label_size:
        #     image_ids.extend(class_img_ids)
        # else:
        #     class_img_ids = class_img_ids[:label_size]
        #     image_ids.extend(class_img_ids)
    # # Remove duplicates
    # image_ids = list(set(image_ids))
    print(RIs)

    print('Finish process.')
