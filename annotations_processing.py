import pandas as pd
import glob
import os
import csv
import ast


# http://www.robots.ox.ac.uk/~vgg/software/via/via.html
# https://courses.cs.washington.edu/courses/cse140/13wi/csv-parsing.html


def read_csv(file_path):
    result = []
    with open(file_path, "r") as read_file:
        reader = csv.reader(read_file)
        for item in reader:
            result.append(item)
    return result


def save_data(data, path):
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(path, index=False, header=False)


# sanity-check of list (1. no duplicate item, 2. correct amount of region_count)
def sanity_check_list(csv_list):
    gp_name = csv_list[1][0]
    shape = csv_list[1][5]
    for row in range(2, len(csv_list)):
        if csv_list[row][0] == gp_name:
            if csv_list[row][5] == shape:
                print('duplicate')
        else:
            gp_name = csv_list[row][0]
            shape = csv_list[row][5]
    print('Checked 1.')

    gp_name = csv_list[1][0]
    count = 1
    for row in range(2, len(csv_list)):
        if csv_list[row][0] == gp_name:
            count += 1
        else:
            for num in range(count):
                prev = num + 1
                if csv_list[row - prev][3] != count:
                    print('incorrect region count')
            gp_name = csv_list[row][0]
            count = 1
    print('Checked 2.')


def construct_anno_csv(csv_list, data_folder, save_path):
    # process filename & file_size
    for row in range(len(csv_list) - 1):
        new_name = ((csv_list[row + 1][0]).split('/'))[-1]
        csv_list[row + 1][0] = new_name
        csv_list[row + 1][1] = os.path.getsize(os.path.join(data_folder, new_name))

    # # see changes
    # for row in range(len(csv_list)):
    #     print(csv_list[row][0], csv_list[row][1])

    # process region_count
    name = ''
    count = 0
    region_count = 0
    region_id = 0
    for row in range(1, len(csv_list)):
        if count == 0:
            name = csv_list[row][0]
            count += 1
            region_count += 1
        else:
            current_name = csv_list[row][0]
            if current_name == name:
                count += 1
                region_count += 1
            else:
                while count > 0:
                    csv_list[row - count][3] = region_count
                    csv_list[row - count][4] = region_id
                    count -= 1
                    region_id += 1

                name = csv_list[row][0]
                count = 1
                region_count = 1
                region_id = 0

    # # see changes
    # for row in range(len(csv_list)):
    #     print(csv_list[row][0], csv_list[row][3], csv_list[row][4], csv_list[row][6])

    # # sanity check of list
    # sanity_check_list(csv_list)

    # save to dst file path
    save_data(csv_list, save_path)
    print()


def filter_anno_csv(csv_file, save_path):
    filter_word = 'null'
    csv_list = read_csv(csv_file)
    new_annos = [csv_list[0]]
    filenames = []
    for i in range(len(csv_list) - 1):
        row = csv_list[i + 1]
        filename = (row[0].split('/'))[-1]
        # print(filename)

        if filter_word in row[-2]:
            filenames.append(filename)
        else:
            shape_attr = ast.literal_eval(row[-2])
            x_points = shape_attr['all_points_x']
            y_points = shape_attr['all_points_y']
            # print(x_points)
            # print(y_points)

            if x_points and y_points:
                if (type(x_points[0]) != list) and (type(y_points[0]) != list):
                    new_annos.append(row)
                else:
                    filenames.append(filename)
            else:
                filenames.append(filename)

    print('The amount of annotations removed is', len(filenames))
    print('The amount of annotations left is', (len(new_annos) - 1), 'minus headings')

    # remove filename from filenames list if there exists a valid annotation for that image
    for i in range(len(new_annos) - 1):
        row = new_annos[i + 1]
        filename = (row[0].split('/'))[-1]
        if filename in filenames:
            filenames.remove(filename)

    # remove duplicate filenames
    filenames = list(set(filenames))

    # save new annotations to specified path
    save_data(new_annos, save_path)

    return new_annos, filenames


def count_cat_data(annotations, categories, return_count=False):
    total = 0
    count_dict = {}
    exceptions = []
    for category in categories:
        amount = 0
        for annotation in annotations[1:]:  # ignore headings
            attr = ast.literal_eval(annotation[-1])
            cat = attr['category']
            if cat == category:
                amount += 1
            if cat not in categories:
                exceptions.append(annotation)
        print('Category ' + category + ' contains ' + str(amount) + ' in annotations.')
        if return_count is True:
            count_dict.update([(category, amount)])
        total += amount
    print('These categories are ' + str(total) + ' in total.')
    if exceptions:
        for exception in exceptions:
            print('Exception:', exception)

    if return_count is True:
        return count_dict


def clean_data(filenames, data_folder, more_files=[]):
    images = glob.glob(data_folder + '/*.jpg')
    all_file = []
    for img in images:
        all_file.append(os.path.basename(img))
    print('The amount of images in data folder is:', len(all_file))
    print('The amount of images to be removed is:', len(filenames))

    amount = 0
    invalid_img = []
    for filename in filenames:
        rm_path = os.path.join(data_folder, filename)
        try:
            os.remove(rm_path)
            amount += 1
        except:
            invalid_img.append(filename)

    print('Successfully removed ' + str(amount) + ' images in dir "' + data_folder + '".')

    if more_files:
        for extra_path in more_files:
            try:
                os.remove(extra_path)
            except:
                print('Failed to delete extra file:', extra_path)
        print('Finish removing extra files.')

    if invalid_img:
        for img in invalid_img:
            print('Image not in data folder:', img)

    images_now = glob.glob(data_folder + '/*.jpg')
    files_now = []
    for img in images_now:
        files_now.append(os.path.basename(img))
    print('The amount of files in data folder now is:', len(files_now))


# filter the csv file and save those only present in our dataset with our interested categories
categories = ['dog', 'cake', 'bed']
keys = ['filename', 'file_size', 'file_attributes', 'region_count', 'region_id', 'region_shape_attributes',
        'region_attributes']

if __name__ == '__main__':
    # variables
    mode = input('Enter mode (select/filter): ')
    image_dir = input('Enter image directory (e.g., ./cocoDS/val2017): ')
    csv_file = input('Enter load path for the csv file (e.g., ./new_annotations/val/via_export_csv.csv): ')
    save_csv = input('Enter save path for the new csv (e.g., ./new_annotations/val/filter_anns.csv): ')

    # run different process depends on mode
    if mode == 'select':
        # get all the filenames of our dataset
        images = glob.glob(image_dir + '/*.jpg')
        all_file = []
        for img in images:
            all_file.append(os.path.basename(img))

        # read csv file as dictionary and select interested annotations (csv file groups annotations of same filename)
        file_list = read_csv(csv_file)
        annotations = [file_list[0]]
        for i in range(len(file_list) - 1):
            current_row = file_list[i + 1]
            filename = (current_row[0].split('/'))[-1]
            try:
                category = (current_row[-1].split(':'))[-1].split('}')[0].split('"')[1]
                # print(category)
            except IndexError:
                category = current_row[-1]
                # print(str(i + 2) + ' in csv file is ', category)
            if filename in all_file and category in categories:
                annotations.append(current_row)

        # # our interested annotations
        # print(len(annotations))     # headings (1) + annotations (697) = 698
        # print(annotations[0], '\n', annotations[1])

        # process filename, file_size and region_count
        construct_anno_csv(annotations, image_dir, save_csv)
    elif mode == 'filter':
        do_clean = input('Do you want to delete unqualified images? (y/n): ')
        if do_clean == 'y':
            extra_files = []
            while True:
                extra = input('Enter path to non-image to be deleted deleted file or no to stop '
                              '(e.g., ./cocoDS/val2017/annotation_val.csv): ')
                if extra == 'no':
                    break
                extra_files.append(extra)

        # filter and get new annotations
        anns, fns = filter_anno_csv(csv_file, save_csv)

        # count the number of data for each category
        count_cat_data(anns, categories)

        # delete unqualified images
        if do_clean == 'y':
            clean_data(fns, image_dir, extra_files)
            print('Finish removing.')
    else:
        print('Invalid mode!')

    print('Finish process.')
