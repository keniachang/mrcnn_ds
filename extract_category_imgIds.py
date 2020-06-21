"""
The dataset to test dog accuracy on Network B shall only contain dog images (no mix of other categories).
Retrieve the dataset's image ids from the COCO val 2014 dataset
"""

from PythonAPI.pycocotools.coco import COCO


def get_specified_category_image_ids(coco, cat_id, size=None):
    image_ids = list(coco.getImgIds(catIds=[cat_id]))
    image_ids = list(set(image_ids))

    # All classes
    class_ids = sorted(coco.getCatIds())

    category_only_image_ids = []
    # check annotation(s) of each image to make sure only dog
    for img_id in image_ids:
        # print('\nImage', img_id, ':')
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=[img_id], catIds=class_ids, iscrowd=None))

        next_img = False
        for annotation in annotations:
            # print(annotation)
            # print(annotation['category_id'])
            if annotation['category_id'] != cat_id:
                next_img = True
                # print('NEXT!')
                break

        if not next_img:
            # print('Pass')
            category_only_image_ids.append(img_id)

        if len(category_only_image_ids) == size:
            # print('Meet required amount.')
            break

    return category_only_image_ids


if __name__ == '__main__':
    # ann_path = "../drive/My Drive/coco_datasets/annotations/instances_val2014.json"
    ann_file = "./cocoDS/annotations/instances_val2014.json"
    dog_id = 18
    dataset_size = 2

    coco_val = COCO(ann_file)
    pure_dog_dataset_img_ids = get_specified_category_image_ids(coco_val, dog_id, dataset_size)
    print(pure_dog_dataset_img_ids)

    # show dataset's images OR enter the image id on http://cocodataset.org/#explore to see the image & its annotation
    for image_id in pure_dog_dataset_img_ids:
        image_url = coco_val.imgs[image_id]['coco_url']
        print(image_url)

    print('\nFinish process.')
