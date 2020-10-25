import paddle.fluid as fluid
import os
from basic_data_preprocessing import TrainAugmentation
from paddle.fluid.dygraph import to_variable
from basic_dataloader import BasicDataLoader
from basic_model import BasicModel
from PIL import Image
import numpy as np


def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color


def save_blend_image(image_file, pred_file):
    image1 = Image.open(image_file)
    image2 = Image.open(pred_file)
    image1 = image1.convert('RGBA')
    image2 = image2.convert('RGBA')
    image = Image.blend(image1, image2, 0.5)
    o_file = pred_file[0:-4] + "_blend.png"
    image.save(o_file)


def inference_resize():
    pass


def inference_sliding():
    pass


def inference_multi_scale():
    pass


def save_images():
    pass


# this inference code reads a list of image path, and do prediction for each image one by one
def main():
    # 0. env preparation
    place = fluid.CPUPlace()
    infer_exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()
    param_dirname = "G:/python_workspace/test_paddle/day03/best_model"
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(param_dirname, infer_exe)
        # 1. create model
        model = BasicModel()

        # 2. load pretrained model

        # 3. read test image list
        data_list = []
        image_folder = "./dummy_data"
        image_list_file = "./dummy_data/list.txt"
        with open(image_list_file) as f:
            for line in f:
                data_path = os.path.join(image_folder, line.split()[0])
                label_path = os.path.join(image_folder, line.split()[1])
                data_list.append((data_path, label_path))

        # 4. create transforms for test image, transform should be same as training
        augment = TrainAugmentation(image_size=256)

        basic_dataloader = BasicDataLoader(image_folder=image_folder,
                                           image_list_file=image_list_file,
                                           transform=augment,
                                           shuffle=True)

        # 5. loop over list of images
        for idx, (data, label) in enumerate(basic_dataloader):
            # 6. read image and do preprocessing

            # 7. image to variable
            image = to_variable(data)

            # 8. call inference func
            results = infer_exe.run(inference_program,
                                    feed={feed_target_names[0]: np.array(image)},
                                    fetch_list=fetch_targets)
            # 9. save results
            # save_images()



if __name__ == "__main__":
    main()
