import os
from PythonAPI.pycocotools.coco import COCO
import requests
from find_catID import coco_categories
from extract_category_imgIds import get_specified_category_image_ids
import skimage.io
import matplotlib.pyplot as plt


# https://github.com/cocodataset/cocoapi/issues/271

if __name__ == '__main__':
    # # download the images of a specific category of the coco dataset
    # dataset = input('Enter the dataset to be downloaded (train2017/val2017): ')
    # category = input('Enter the category to be downloaded (dog, cake, bed): ')
    #
    # data_folder_path = './cocoDS/' + dataset + '/'
    # os.makedirs(data_folder_path, exist_ok=True)
    #
    # ann_file = './cocoDS/annotations/instances_' + dataset + '.json'
    # coco = COCO(ann_file)
    # # cats = coco.loadCats(coco.getCatIds())
    # # nms = [cat['name'] for cat in cats]
    # # print('COCO categories: \n{}\n'.format(' '.join(nms)))
    #
    # # trainval2017 - 'dog' (4562), 'cake' (3049), 'bed' (3831)
    # # 'dog' - train=4385 & test=177
    # # 'cake' - train=2925?/2859? & test=124?/123?   (# img downloaded)?/(# in folder)?
    # # 'bed' - train=3682?/3319? & test=149?/131?
    # # totoal train of 3 categories = 4385+2859+3319=10563
    # # totoal val of 3 categories = 4385+2859+3319=431
    # # 11442 - 10994 = 448 images are absent
    # catIds = coco.getCatIds(catNms=[category])
    # imgIds = coco.getImgIds(catIds=catIds)
    # images = coco.loadImgs(imgIds)
    # # print("imgIds: ", imgIds)
    # # print("images: ", images)
    #
    # i = 0
    # for im in images:
    #     # print("im: ", im)
    #
    #     img_data = requests.get(im['coco_url']).content
    #
    #     with open(data_folder_path + im['file_name'], 'wb') as handler:
    #         handler.write(img_data)
    #
    #     i += 1
    #     print('Images Downloaded: ' + str(i))

    # # download pure dog imgaes from val2014 for Networks A, B, C for evaluation
    # # save_data_dir = './coco_val2014_pure_dog_images/'
    # coco = COCO('./cocoDS/annotations/instances_train2014.json')
    # print()
    #
    # # download images
    # selected_cat_id0 = 18   # dog
    # image_ids = get_specified_category_image_ids(coco, selected_cat_id0)
    # print(image_ids)
    # i = 0
    # for img_id in image_ids:
    #
    #     img_data = requests.get(coco.imgs[img_id]['coco_url']).content
    #
    #     with open(save_data_dir + coco.imgs[img_id]['file_name'], 'wb') as handler:
    #         handler.write(img_data)
    #
    #     i += 1
    #     print('Images Downloaded: ' + str(i))
    #
    # print('\nFinish downloading pure dog images.')

    # # test to access the pure dog data
    # image_dir = './coco_val2014_pure_dog_images'
    # for i in image_ids:
    #     path = os.path.join(image_dir, coco.imgs[i]['file_name'])
    #     try:
    #         image = skimage.io.imread(path)
    #     except:
    #         image = []
    #         print('Do not use this image and its annotation for training/testing!')
    #     # print(image)
    #
    #     print('image id:', i)
    #     print('filename:', coco.imgs[i]['file_name'])
    #
    #     # Draw image
    #     plt.axis('off')
    #     plt.imshow(image)
    #     plt.show()

    # size of each label pure data in train2014
    coco = COCO('./cocoDS/annotations/instances_train2014.json')
    print()
    qualified_size = 200
    qualified_labels = []
    minimum_size = qualified_size
    label = ''
    for category in coco_categories:
        class_id = (coco.getCatIds(catNms=[category]))[0]

        image_ids = get_specified_category_image_ids(coco, class_id)
        subset_size = len(image_ids)
        print(category, subset_size, '\n')

        if subset_size < minimum_size:
            minimum_size = subset_size
            label = category
        elif subset_size >= qualified_size:
            qualified_labels.append(category)

    print('Amount of label with qualified size:', len(qualified_labels))
    print('Qualified labels:', qualified_labels)
    print('Smallest:', label, '=', minimum_size)
