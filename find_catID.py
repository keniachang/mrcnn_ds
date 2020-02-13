from PythonAPI.pycocotools.coco import COCO
# import json

# coco categories names
coco_categories = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow',
                   'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                   'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                   'skis', 'snowboard', 'sports', 'ball', 'kite',
                   'baseball', 'bat', 'baseball glove', 'skateboard', 'surfboard',
                   'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                   'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                   'pizza', 'donut', 'cake', 'chair', 'couch',
                   'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                   'laptop', 'mouse', 'remote keyboard', 'cell phone', 'microwave oven',
                   'toaster', 'sink', 'refrigerator', 'book', 'clock',
                   'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

dataset_dir = './cocoDS'
year = '2017'
dataset = input('Enter the dataset (train2017/val2017): ')
categories = []  # ['dog', 'cake', 'bed']
while True:
    category = input('Enter the category name (dog/cake/bed), or no to stop: ')
    if category in coco_categories:
        categories.append(category)
    elif category == 'no':
        break
    else:
        print('Unknown category has been entered!')

# data_folder_path = './cocoDS/annotations_subset/'
# os.makedirs(data_folder_path, exist_ok=True)

if dataset == 'train2017':
    subset = 'train'
elif dataset == 'val2017':
    subset = 'val'
else:
    print('Invalid dataset was entered!')
    exit(-1)

coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))

class_ids = []
for category in categories:
    class_ids.extend(list(coco.getCatIds(catNms=[category])))
print(categories, class_ids)

# image_ids = []
# for class_id in class_ids:
#     image_ids.extend(list(coco.getImgIds(catIds=[class_id])))
# # Remove duplicates
# image_ids = list(set(image_ids))
