import numpy as np
import scipy.stats as ss
import pandas as pd
import skimage.io
import skimage.color
import skimage.transform
import random
from distutils.version import LooseVersion
import matplotlib.pyplot as plt
from PIL import Image


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


def mold_inputs(config, image):
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
    # Resize image
    molded_image, window, scale, padding, crop = resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    molded_image = mold_image(molded_image, config)
    return molded_image


def mask_image(coco_obj, img_id, cat_ids, image, mask_mode):
    anns = coco_obj.loadAnns(coco_obj.getAnnIds(imgIds=[img_id], catIds=cat_ids, iscrowd=None))
    mask = coco.annToMask(anns[0])
    for ind in range(len(anns)):
        mask |= coco.annToMask(anns[ind])
    if mask_mode == 'binary_mask':
        return mask     # binary mask
    else:
        masked_image = np.zeros(shape=image.shape[:3], dtype=np.uint8)
        for h in range(image.shape[0]):
            for w in range(image.shape[1]):
                if mask[h, w] != 0:
                    masked_image[h, w, 0] = image[h, w, 0]
                    masked_image[h, w, 1] = image[h, w, 1]
                    masked_image[h, w, 2] = image[h, w, 2]
        return masked_image   # image


if __name__ == '__main__':
    from PythonAPI.pycocotools.coco import COCO
    import train_xyL_bus as train_net   # where the CocoConfig from
    from find_catID import coco_categories

    # get the coco labels used in training x, y labels network to calculate RI
    import argparse
    parser = argparse.ArgumentParser(description='Calculate KL_div on a network.')
    parser.add_argument('--label', required=True,
                        metavar="a coco label name or self defined label",
                        help="The label of the training dataset to be computed RI on.")
    parser.add_argument('--mode', required=True,
                        metavar="<original|mask_image|binary_mask>",
                        help="The label of the network to be computed RI on.")
    args = parser.parse_args()

    """
    For self-defined label:
    """
    # assert args.label == 'bus' or args.label == 'train' or args.label == 'mix67'

    """
    For coco label:
    """
    assert args.label in coco_categories[:10]

    assert args.mode == 'original' or args.mode == 'mask_image' or args.mode == 'binary_mask'

    label = args.label
    mode = args.mode

    """
    For self-defined label:
    """
    # if label == 'mix67':
    #     labels = ['bus', 'train']
    #     label_size = 250
    # else:
    #     labels = [label]
    #     label_size = 500

    """
    For coco label:
    """
    labels = [label]
    label_size = 500

    # config
    netL_config = train_net.CocoConfig()
    netL_config.IMAGE_MIN_DIM = 400
    netL_config.IMAGE_MAX_DIM = 512
    netL_config.DEFAULT_LOGS_DIR = './logs'
    # netL_config.display()

    coco = COCO("../drive/My Drive/coco_datasets/annotations/instances_train2014.json")
    print()

    # get the image ids of the training dataset
    class_ids = coco.getCatIds(catNms=labels)

    """
    For self-defined label: Get the training dataset / image ids
    """
    # class_img_ids = []
    # if len(class_ids) == 1:
    #     cat_img_ids = (train_net.get_xy_labels_data(coco, class_ids[0], label_size))[0]
    #     class_img_ids.extend(cat_img_ids)
    # else:
    #     cat_img_ids = train_net.get_xy_labels_data(coco, class_ids[0], (label_size * 2), class_ids[1])
    #     cat1_img_ids = cat_img_ids[0][:label_size]
    #     cat2_img_ids = cat_img_ids[1][:label_size]
    #     class_img_ids.extend(cat1_img_ids)
    #     class_img_ids.extend(cat2_img_ids)

    """
    For coco label: Get the training dataset / first 500 image ids
    """
    class_img_ids = list(coco.getImgIds(catIds=class_ids))
    class_img_ids = list(set(class_img_ids))
    class_img_ids = class_img_ids[:label_size]

    """
    Following code same for both ways
    """
    print(labels, 'training size:', len(class_img_ids))

    # get the images & also resize them as how it was prep for training in mrcnn
    num = 0
    class_images = []
    print('Loading and resizing images...')
    for i in class_img_ids:
        img_path = coco.imgs[i]['coco_url']
        img = load_image(img_path)  # (height, width, 3)

        if mode != 'original':
            # mask the image or get the binary mask
            m_img = mask_image(coco, i, class_ids, img, mode)
            # plt.imshow(m_img)
            # plt.axis('off')
            # plt.show()

        # resize the image
        if mode == 'original':
            molded_img = mold_inputs(netL_config, img)
        elif mode == 'mask_image':
            molded_img = mold_inputs(netL_config, m_img)
        else:
            molded_img = skimage.transform.resize(m_img, (512, 512))

        class_images.append(molded_img)
        num += 1
        if num % netL_config.STEPS_PER_EPOCH == 0:
            print(label + ':', num, 'images are loaded and resized to shape', molded_img.shape)

    print('\nDataset shape:', np.shape(class_images))
    print('Images are loaded and resized, calculating RI...')
    class_ri = relative_information(class_images)
    print(label, 'RI:', class_ri)

    # for i, img_id in enumerate(class_img_ids):
    #     img_path = coco.imgs[img_id]['coco_url']
    #     print(i, img_path)

    print('\nFinish process.')
