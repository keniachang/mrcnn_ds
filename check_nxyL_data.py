from PythonAPI.pycocotools.coco import COCO


network_cats = ['person', 'bicycle']
subset = 'train'
label_size = 250
images_per_weight = 10
coco = COCO("./cocoDS/annotations/instances_train2014.json")

selected_class_ids = coco.getCatIds(catNms=network_cats)
image_ids = []
if subset == 'train':
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

print(subset, 'size:', len(image_ids))
if subset == 'train' and len(image_ids) % images_per_weight != 0:
    print('STOP: training dataset is not a multiple of 10 which will cause some samples to be used in '
          'training for more than once!')
    exit(1)
