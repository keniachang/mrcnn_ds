import calculate_networks_KL as n_KL
from find_catID import coco_categories
from PythonAPI.pycocotools.coco import COCO


def relative_information(dataset):
    s = n_KL.resize_layer(sum(dataset) / len(dataset))
    ri = 0
    for si in dataset:
        s_flat = n_KL.resize_layer(si)
        ri = ri + n_KL.KL_div(s_flat, s) + n_KL.KL_div(s, s_flat)
    return ri


if __name__ == '__main__':
    label_size = 500
    labels = coco_categories[:10]

    # calculate RI for each coco label (first 10 coco labels) using each label's 500 training images
    coco = COCO("../drive/My Drive/coco_datasets/annotations/instances_train2014.json")

    class_ids = coco.getCatIds(catNms=labels)
    image_ids = []
    for class_id in class_ids:
        class_img_ids = list(coco.getImgIds(catIds=[class_id]))
        class_img_ids = list(set(class_img_ids))
        print(len(class_img_ids))
        # class_img_ids = class_img_ids[:label_size]

        # # TODO: get the images & also resize them as how it was prep for training
        # class_data = class_img_ids
        # class_ri = relative_information(class_data)

        # # print current coco label RI
        # category = coco.loadCats(class_id)
        # label_name = category['name']
        # print(label_name, class_ri)

        # # get all labels' training data
        # if len(class_img_ids) < label_size:
        #     image_ids.extend(class_img_ids)
        # else:
        #     class_img_ids = class_img_ids[:label_size]
        #     image_ids.extend(class_img_ids)
    # # Remove duplicates
    # image_ids = list(set(image_ids))
