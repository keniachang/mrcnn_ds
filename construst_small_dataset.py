"""
Sample Usage 1: python construst_small_dataset.py --mode=small --dataPath=./coco/ --dataset=train2017 --size=400 --load=./new_annotations/train/filter_anns.csv --copyDir=./cocoDS/train2017/

Sample Usage 2: python construst_small_dataset.py --mode=small --dataPath=./coco/ --dataset=val2017 --size=200 --load=./new_annotations/val/filter_anns.csv --copyDir=./cocoDS/val2017/

Sample Usage 3: python construst_small_dataset.py --mode=small --dataPath=./coco/ --dataset=samples --size=30 --load=./coco/train2017/subset_anns.csv --copyDir=./coco/train2017/

Sample Usage 4: python construst_small_dataset.py --mode=dark --dataPath=./coco/ --dataset=dark --load=./coco/samples/subset_anns.csv --copyDir=./coco/samples/
"""

from annotations_processing import read_csv, save_data, count_cat_data, categories
import ast
import pathlib
import os
import shutil
import glob
import skimage.io
import numpy as np
import cv2

# # construct data of one category
# categories = ['dog']


def check_data_dir(path, data_folder):
    dataset_path = path + data_folder
    pathlib.Path(dataset_path).mkdir(parents=True, exist_ok=True)


def construct_base_dataset(csv_file, classes, size, save_filenames_path, save_imgs_path, copy_imgs_from, save_csv):
    # cat_size = round(size / len(classes)) + 1
    cat_size = round(size / len(classes))
    print('Each category will have', cat_size, 'annotations for this constructed dataset.')
    annotations = read_csv(csv_file)
    print('Total amount of annotations is', (len(annotations) - 1), 'in specified annotation csv file.')

    new_anns = [annotations[0]]
    filenames = []
    cat_counts = count_cat_data(annotations, classes, return_count=True)
    for cat in classes:
        if cat_size > cat_counts[cat]:
            print('Category', cat, 'can only contains', cat_counts[cat], 'annotations.')
        else:
            cat_counts[cat] = cat_size

        amount = 0
        for ann in annotations[1:]:
            attr = ast.literal_eval(ann[-1])
            if attr['category'] == cat:
                new_anns.append(ann)
                filenames.append(ann[0])
                amount += 1
            if amount == cat_counts[cat]:
                break
    print(cat_counts)
    print('Final total amount of annotations is', len(filenames), 'for this constructed dataset.')

    # remove duplicates
    filenames = list(set(filenames))

    # Create the dataset with images in specified path
    save_data(filenames, save_filenames_path)
    print('The constructed dataset contains', len(filenames), 'images.')

    if save_imgs_path is not None:
        for filename in filenames:
            src = copy_imgs_from + filename
            src = os.path.abspath(src)
            try:
                shutil.copy(src, save_imgs_path)
            except:
                print('Failed to copy', src)
        print('Images are saved in', save_imgs_path, 'and can be proceed.')

    if save_path is not None:
        save_data(new_anns, save_csv)
        print('The constructed dataset annotation is saved as', save_csv)

    return new_anns, filenames


def construct_dark_dataset(csv_file, save_imgs_path, copy_imgs_from, save_csv):
    annotations = read_csv(csv_file)
    print('Total amount of annotations is', (len(annotations) - 1), 'in specified annotation csv file.')

    filenames = glob.glob(copy_imgs_from + '*.jpg')
    imgs_names = []
    for filename in filenames:
        imgs_names.append(os.path.basename(filename))

    print('The constructed dataset contains', len(filenames), 'images.')

    if save_imgs_path is not None:
        # get every image and turn it dark, then save it
        for img_name in imgs_names:
            src = copy_imgs_from + img_name
            dst = save_imgs_path + img_name

            image = skimage.io.imread(src)
            dark_img = np.zeros(image.shape, np.uint8)
            cv2.imshow(('Dark image of ' + img_name), dark_img)
            cv2.imwrite(dst, dark_img)
        print('Images are processed and saved in', save_imgs_path)

    if save_path is not None:
        save_data(annotations, save_csv)
        print('The constructed dataset annotation is saved as', save_csv)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Construct the dataset used for experiment.')
    parser.add_argument('--mode', required=True,
                        metavar="<small|dark>",
                        help="Type of dataset to be constructed.")
    parser.add_argument('--dataPath', required=True,
                        metavar="path/to/folder/save/data/",
                        help="Directory of the constructed dataset to be saved, such as ./coco/")
    parser.add_argument('--dataset', required=True,
                        metavar="<constructed_dataset_folder>",
                        help="The folder name saving the data of the constructed dataset, such as train2017, val2017, "
                             "samples, or dark")
    parser.add_argument('--size', required=False,
                        default=100,
                        metavar="<size>",
                        help="Size of the constructed dataset, each category takes equal amount if possible",
                        type=int)
    parser.add_argument('--load', required=True,
                        metavar="path/to/load/ann.csv",
                        help="Path to the annotation csv file to be loaded, "
                             "such as ./new_annotations/train/filter_anns.csv")
    parser.add_argument('--saveImg', required=False,
                        default='y',
                        metavar="<y|n>",
                        help='Save images for the constructed dataset (y/n) (default=y)')
    parser.add_argument('--copyDir', required=True,
                        metavar="path/to/src/data/",
                        help='Directory of the dataset containing the data to be used, such as ./cocoDS/train2017/')
    parser.add_argument('--saveAnn', required=False,
                        default='y',
                        metavar="<y|n>",
                        help='Save annotation csv file for the constructed dataset (y/n) (default=y)')
    args = parser.parse_args()

    mode = args.mode
    data_path = args.dataPath
    dataset = args.dataset
    dataset_size = args.size
    load_path = args.load
    create_imgs = args.saveImg
    copy_imgs_path = args.copyDir
    do_save = args.saveAnn

    print('\n****************************************')
    print('Start process...')
    print('****************************************')

    assert mode in ['small', 'dark']
    assert len(categories) > 0
    assert dataset_size >= len(categories)
    assert create_imgs in ['y', 'n']
    assert do_save in ['y', 'n']

    if create_imgs == 'y':
        imgs_path = data_path + dataset + '/'
    else:
        imgs_path = None

    save_path = data_path + dataset + '/' + 'subset_anns.csv'
    if do_save == 'n':
        save_path = None

    if mode == 'small':
        # train: 402 annotations, 281 images
        # val: 204 annotations, 137 imgaes
        # samples: 30 annotations, 19 images

        check_data_dir(data_path, dataset)
        save_fns_path = data_path + dataset + '_images.csv'
        anns, fns = construct_base_dataset(load_path, categories, dataset_size, save_fns_path,
                                           imgs_path, copy_imgs_path, save_path)
    elif mode == 'dark':
        # dark
        check_data_dir(data_path, dataset)
        construct_dark_dataset(load_path, imgs_path, copy_imgs_path, save_path)
    else:
        print('Invalid mode!')

    print('****************************************')
    print('Finish process.')
    print('****************************************\n')
