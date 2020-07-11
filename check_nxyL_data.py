from PythonAPI.pycocotools.coco import COCO
from extract_category_imgIds import get_xy_exclude_img_ids as get_xy_labels_data

if __name__ == '__main__':
    label = 'mix'
    assert label == 'airplane' or label == 'bus' or label == 'mix'
    if label == 'mix':
        label_size = 250
    else:
        label_size = 500
    if label == 'mix':
        network_labels = ['airplane', 'bus']
    else:
        network_labels = [label]
    print(network_labels)

    network_cats = network_labels
    subset = 'train'
    images_per_weight = 10
    coco = COCO("./cocoDS/annotations/instances_train2014.json")

    selected_class_ids = coco.getCatIds(catNms=network_cats)
    image_ids = []
    if subset == 'train':
    #     for cat_id in selected_class_ids:
    #         cat_img_ids = list(coco.getImgIds(catIds=[cat_id]))
    #         cat_img_ids = list(set(cat_img_ids))
    #         if len(cat_img_ids) < label_size:
    #             image_ids.extend(cat_img_ids)
    #         else:
    #             cat_img_ids = cat_img_ids[:label_size]
    #             image_ids.extend(cat_img_ids)
        if len(selected_class_ids) == 1:
            cat_id = selected_class_ids[0]
            cat_img_ids = (get_xy_labels_data(coco, cat_id, label_size))[0]
            image_ids.extend(cat_img_ids)
        else:
            cat_id1 = selected_class_ids[0]
            cat_id2 = selected_class_ids[1]
            cat_img_ids = get_xy_labels_data(coco, cat_id1, (label_size*2), cat_id2)
            cat1_img_ids = cat_img_ids[0][:label_size]
            cat2_img_ids = cat_img_ids[1][:label_size]
            image_ids.extend(cat1_img_ids)
            image_ids.extend(cat2_img_ids)
    # Remove duplicates
    image_ids = list(set(image_ids))

    print(subset, 'size:', len(image_ids))
    if subset == 'train' and len(image_ids) % images_per_weight != 0:
        print('STOP: training dataset is not a multiple of 10 which will cause some samples to be used in '
              'training for more than once!')
        exit(1)
