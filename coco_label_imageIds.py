import pandas as pd
import csv
from PythonAPI.pycocotools.coco import COCO


def save_label_image_ids(coco, cat_id, size, save_path):
    image_ids = []
    image_ids.extend(list(coco.getImgIds(catIds=cat_id)))
    image_ids = list(set(image_ids))

    if len(image_ids) >= size:
        image_ids = image_ids[:size]

    data = pd.DataFrame(image_ids)
    data.to_csv(save_path, index=False, header=False)

    return image_ids


def read_label_image_ids(path):
    image_ids = []
    with open(path, "r") as read_file:
        reader = csv.reader(read_file)
        for image_id in reader:
            image_ids.append(int(image_id))

    return image_ids


def get_label_image_ids_file_path(label, size, file_dir='./networks_labels_imgIds'):
    file_path = file_dir + 'coco_' + label + str(size) + 'imgIds.csv'
    return file_path


if __name__ == '__main__':
    pass
