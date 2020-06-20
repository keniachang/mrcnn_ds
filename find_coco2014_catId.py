from PythonAPI.pycocotools.coco import COCO

coco = COCO("cocoDS/annotations/instances_train2014.json")
find_categories = ['car', 'dog', 'cake']

found_catIds = coco.getCatIds(catNms=find_categories)
print(found_catIds)
