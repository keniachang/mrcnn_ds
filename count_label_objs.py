if __name__ == '__main__':
    from PythonAPI.pycocotools.coco import COCO
    from extract_category_imgIds import get_xy_exclude_img_ids as get_xy_labels_data

    # import argparse
    # parser = argparse.ArgumentParser(description='Calculate KL_div on a network.')
    # parser.add_argument('--label', required=True,
    #                     metavar="<bus|train|mix67>",
    #                     help="The label of the network to be computed KL_div on.")
    # args = parser.parse_args()
    # assert args.label == 'bus' or args.label == 'train' or args.label == 'mix67'
    #
    # label = args.label
    label = 'bus'
    if label == 'mix67':
        labels = ['bus', 'train']
        label_size = 250
    else:
        labels = [label]
        label_size = 500

    # ann_file = "../drive/My Drive/coco_datasets/annotations/instances_train2014.json"
    ann_file = "./cocoDS/annotations/instances_train2014.json"

    coco = COCO(ann_file)
    print()

    cat_ids = coco.getCatIds(catNms=labels)
    if label == 'mix67':
        cat_img_ids = get_xy_labels_data(coco, cat_ids[0], (label_size * 2), cat_ids[1])

        dataset_size = 0
        objects = 0
        for index, c_img_ids in enumerate(cat_img_ids):
            cat_objs = 0
            cc_img_ids = c_img_ids[:label_size]
            dataset_size += len(cc_img_ids)
            for i in cc_img_ids:
                annotations = coco.loadAnns(coco.getAnnIds(imgIds=[i], catIds=cat_ids, iscrowd=None))
                objs = len(annotations)
                cat_objs += objs
            print(label, 'training dataset has', cat_objs, labels[index], 'objects')
            objects += cat_objs
    else:
        cat_img_ids = (get_xy_labels_data(coco, cat_ids[0], label_size))[0]
        # cat_img_ids = [230893, 167033, 22274, 493936]     # 2, 2, 1, 4 bus objects in each image id having bus

        dataset_size = len(cat_img_ids)
        objects = 0
        # dataset_anns = []
        for i in cat_img_ids:
            annotations = coco.loadAnns(coco.getAnnIds(imgIds=[i], catIds=cat_ids, iscrowd=None))
            # print(annotations)
            objs = len(annotations)
            # print(objs)
            objects += objs
            # dataset_anns.append(annotations)

    print(label, 'training dataset size:', dataset_size)
    print(label, 'has', objects, 'total labeled objects in the training dataset specifically for the x, y network')

    # for i, img_id in enumerate(image_ids):
    #     img_path = coco.imgs[img_id]['coco_url']
    #     print(i, img_path)

    print('\nFinish process.')
