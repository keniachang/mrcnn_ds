from annotations_processing import read_csv, save_data, count_cat_data, categories
import ast
import os
import shutil


def construct_base_dataset(csv_file, classes, subset, size, save_imgs, save_csv):
    cat_size = round(size / len(classes)) + 1
    print('Each category will contain', cat_size, 'annotations.')
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
    print('Total amount of annotations is', len(filenames), 'for this constructed dataset.')

    # remove duplicates
    filenames = list(set(filenames))

    # Create the dataset with images in specified path
    if save_imgs is not None:
        for filename in filenames:
            src = './cocoDS/' + subset + '/' + filename
            src = os.path.abspath(src)
            try:
                shutil.copy(src, save_imgs)
            except:
                print('Failed to copy', src)
        print('Images are saved in', save_imgs, 'and can be proceed.')

    if save_path is not None:
        save_data(new_anns, save_csv)
        print('The constructed dataset annotation is saved as', save_csv)

    return new_anns, filenames


def construct_samples_dataset():
    pass


def construct_dark_dataset():
    pass


if __name__ == '__main__':
    # train: 402 annotations, 281 images
    # val: 204 annotations, 137 imgaes
    data_path = './coco/'
    load_file = 'filter_anns.csv'
    dataset = input('Enter which dataset to construct from (train2017/val2017): ')
    dataset_size = int(input('Enter the size of the constructed dataset (400/200): '))
    create_imgs = input('Wanna create images for dataset? (y/n): ')
    do_save = input('Wanna save annotation of constructed dataset? (y/n): ')

    assert create_imgs in ['y', 'n']
    assert do_save in ['y', 'n']
    assert len(categories) > 0
    assert dataset_size >= len(categories)

    assert dataset in ['train2017', 'val2017']
    if dataset == 'train2017':
        load_path = './new_annotations/train/' + load_file
    else:
        load_path = './new_annotations/val/' + load_file
    save_path = data_path + dataset + '/' + 'subset_anns.csv'

    if create_imgs == 'y':
        imgs_path = data_path + dataset + '/'
    else:
        imgs_path = None

    if do_save == 'n':
        save_path = None

    anns, fns = construct_base_dataset(load_path, categories, dataset, dataset_size, imgs_path, save_path)
    print('The constructed dataset contains', len(fns), 'images.')

    if do_save == 'y':
        save_fns_path = data_path + dataset + '_images.csv'
        save_data(fns, save_fns_path)

    print('\nFinish process.')
