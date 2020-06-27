from PythonAPI.pycocotools.coco import COCO
from find_catID import coco_categories

if __name__ == '__main__':
    network_label_num = 10
    assert 0 < network_label_num < 81

    label_size = 500
    coco_labels = coco_categories[:network_label_num]

    coco = COCO('./cocoDS/annotations/instances_train2014.json')
    selected_class_ids = coco.getCatIds(catNms=coco_labels)
    image_ids = []
    for cat_id in selected_class_ids:
        cat_img_ids = list(coco.getImgIds(catIds=[cat_id]))
        cat_img_ids = list(set(cat_img_ids))
        if len(cat_img_ids) < label_size:
            image_ids.extend(cat_img_ids)
        else:
            cat_img_ids = cat_img_ids[:label_size]
            image_ids.extend(cat_img_ids)
    # Remove duplicates
    image_ids = list(set(image_ids))

    # ['person'] subset size: 500
    # ['person', 'bicycle'] subset size: 974
    # ['person', 'bicycle', 'car'] subset size: 1372
    # ['person', 'bicycle', 'car', 'motorcycle'] subset size: 1757
    # ['person', 'bicycle', 'car', 'motorcycle', 'airplane'] subset size: 2223
    # ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus'] subset size: 2555
    # ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train'] subset size: 2999
    # ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck'] subset size: 3245
    # ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat'] subset size: 3675
    # ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
    #  'bus', 'train', 'truck', 'boat', 'traffic light'] subset size: 3851
    print(coco_labels, 'subset size:', len(image_ids))
