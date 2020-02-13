import os
from PythonAPI.pycocotools.coco import COCO
import requests

# https://github.com/cocodataset/cocoapi/issues/271

dataset = input('Enter the dataset to be downloaded (train2017/val2017): ')
category = input('Enter the category to be downloaded (dog, cake, bed): ')

data_folder_path = './cocoDS/' + dataset + '/'
os.makedirs(data_folder_path, exist_ok=True)

ann_file = './cocoDS/annotations/instances_' + dataset + '.json'
coco = COCO(ann_file)
# cats = coco.loadCats(coco.getCatIds())
# nms = [cat['name'] for cat in cats]
# print('COCO categories: \n{}\n'.format(' '.join(nms)))

# trainval2017 - 'dog' (4562), 'cake' (3049), 'bed' (3831)
# 'dog' - train=4385 & test=177
# 'cake' - train=2925?/2859? & test=124?/123?   (# img downloaded)?/(# in folder)?
# 'bed' - train=3682?/3319? & test=149?/131?
# totoal train of 3 categories = 4385+2859+3319=10563
# totoal val of 3 categories = 4385+2859+3319=431
# 11442 - 10994 = 448 images are absent
catIds = coco.getCatIds(catNms=[category])
imgIds = coco.getImgIds(catIds=catIds)
images = coco.loadImgs(imgIds)
# print("imgIds: ", imgIds)
# print("images: ", images)

i = 0
for im in images:
    # print("im: ", im)

    img_data = requests.get(im['coco_url']).content

    with open(data_folder_path + im['file_name'], 'wb') as handler:
        handler.write(img_data)

    i += 1
    print('Images Downloaded: ' + str(i))

print('\nFinish downloading images of ' + category + '.')
