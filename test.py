import sys
import os
import coco

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as mrcnn

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


def load_weight(path, config):
    # Create model
    model1 = mrcnn.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)
    model1.load_weights(path, by_name=True)
    return model1


# ...................................................................................
folder_name = input('Which folder in logs saving the weights? Enter: ')
model_index = int(input('Enter model index (m100 then enter 100): '))
model_path = os.path.join(DEFAULT_LOGS_DIR, folder_name)
model_name = 'mask_rcnn_coco_'
end_model = str(model_index).zfill(4)
end_model = end_model + '.h5'

model_fullname = model_name + end_model
current_model_path = os.path.join(model_path, model_fullname)
print(current_model_path + '\n')
current_model = load_weight(current_model_path, coco.CocoConfig())
