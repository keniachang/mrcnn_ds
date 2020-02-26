from annotations_processing import read_csv, save_data, count_cat_data, categories
import ast
import os
import shutil
import glob

copy_imgs_from = './cocoDS/train2017/'
from_anns_file = './new_annotations/train/filter_anns.csv'
dataset_path = './coco/'
dataset = 'test'
dataset_size = 200 + 200
used_dirs = ['./coco/train2017/', './coco/expand/train2017/']
save_filenames_path = './coco/baseDS_filenames/test_images.csv'
save_imgs_path = dataset_path + dataset + '/'
save_csv = dataset_path + dataset + '/' + 'subset_anns.csv'

cat_size = round(dataset_size / len(categories))
print('Each category will have', cat_size, 'annotations for this constructed dataset.')
annotations = read_csv(from_anns_file)
initial_length = len(annotations)  # including heading

# filter annotations to not include original data
original_fns = []
for used in used_dirs:
    original_images = glob.glob(used + '*.jpg')
    for image_path in original_images:
        bn = os.path.basename(image_path)
        original_fns.append(bn)
    del original_images

filtered_anns = [ann for ann in annotations if ann[0] not in original_fns]
annotations = filtered_anns
del filtered_anns
new_length = len(annotations)
print('Amount of annotations filtered is', (initial_length - new_length), 'and can construct another dataset from',
      (new_length - 1), 'annotations.')

new_anns = [annotations[0]]
filenames = []
cat_counts = count_cat_data(annotations, categories, return_count=True, show_exception=True)
for cat in categories:
    if cat_size > cat_counts[cat]:
        print('Category', cat, 'can only contains', cat_counts[cat], 'annotations.')
    else:
        cat_counts[cat] = cat_size

    amount = 0
    for ann in annotations[1:]:
        attr = ast.literal_eval(ann[-1])
        if attr['category'] == cat:
            new_anns.append(ann)
            filenames.append(ann[0])
            amount += 1
        if amount == cat_counts[cat]:
            break
print(cat_counts)
print('Final total amount of annotations is', len(filenames), 'for this constructed dataset.')

# remove duplicates
filenames = list(set(filenames))

# Create the dataset with images in specified path
save_data(filenames, save_filenames_path)
print('The constructed dataset contains', len(filenames), 'images.')

for filename in filenames:
    src = copy_imgs_from + filename
    src = os.path.abspath(src)
    try:
        shutil.copy(src, save_imgs_path)
    except:
        print('Failed to copy', src)
print('Images are saved in', save_imgs_path, 'and can be proceed.')

save_data(new_anns, save_csv)
print('The constructed dataset annotation is saved as', save_csv)
