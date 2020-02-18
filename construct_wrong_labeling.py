from annotations_processing import read_csv, save_data, categories
import os
import ast
import numpy as np

ann_dir = './coco/samples'
ann_file = 'subset_anns.csv'
shift_max = len(categories)

cat_seq = {}
for i, cat in enumerate(categories):
    num = str(i)
    cat_seq[num] = cat

ann = read_csv(os.path.join(ann_dir, ann_file))

# shifting labeling
for i in range(1, shift_max):
    filename = ann_file.split('.')
    shift_file = filename[0] + '_shift' + str(i) + '.' + filename[1]
    shift_path = os.path.join(ann_dir, shift_file)

    ann = read_csv(os.path.join(ann_dir, ann_file))
    new_ann = [ann[0]]
    for annotation in ann[1:]:
        cat_dict = ast.literal_eval(annotation[-1])
        category = cat_dict['category']
        num = categories.index(category)
        order = (num + i) % shift_max
        label = cat_seq[str(order)]
        cat_dict['category'] = label
        new_cat = str(cat_dict).replace('\'', '"').replace(' ', '')
        # print(num, order, label, new_cat)
        annotation[-1] = new_cat
        new_ann.append(annotation)

    # print(new_ann)
    # print(shift_path)
    save_data(new_ann, shift_path)
    print('Categories are shifted by', i)
    print()

# random labeling
labels = np.random.randint(shift_max, size=(len(ann[1:])))
print(labels)

filename = ann_file.split('.')
shift_file = filename[0] + '_shift_random.' + filename[1]
shift_path = os.path.join(ann_dir, shift_file)

ann = read_csv(os.path.join(ann_dir, ann_file))
new_ann = [ann[0]]
for i, annotation in enumerate(ann[1:]):
    cat_dict = ast.literal_eval(annotation[-1])
    order = labels[i]
    label = cat_seq[str(order)]
    cat_dict['category'] = label
    new_cat = str(cat_dict).replace('\'', '"').replace(' ', '')
    annotation[-1] = new_cat
    new_ann.append(annotation)

# print(new_ann)
# print(shift_path)
save_data(new_ann, shift_path)
print('Categories are shifted randomly.\n')

print('Finish process.')
