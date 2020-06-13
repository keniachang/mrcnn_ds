from PythonAPI.pycocotools.coco import COCO
import skimage.io
import matplotlib.pyplot as plt

coco = COCO("./cocoDS/annotations/instances_train2014.json")
class_ids = coco.getCatIds(catNms=['dog'])
print(class_ids)

image_ids = []
for cat_id in class_ids:
    image_ids.extend(list(coco.getImgIds(catIds=[cat_id])))
# Remove duplicates
image_ids = list(set(image_ids))

# Get 1st image id and its url
image_id0 = image_ids[0]
coco_img_url0 = coco.imgs[image_id0]['coco_url']
print(coco_img_url0)

# # Test reading image from invalid url
# coco_img_url0 = 'http://images.cocodataset.org/train2014/COCO_train2014_002000098304.jpg'
# Read image from url
try:
    image = skimage.io.imread(coco_img_url0)
except:
    image = []
    print('Do not use this image and its annotation for training/testing!')
# print(image)
# Draw image
plt.axis('off')
plt.imshow(image)
plt.show()
