import os
import numpy as np
import cv2
import paddle.fluid as fluid


class BasicDataLoader():
    def __init__(self,
                 image_folder,
                 image_list_file,
                 transform=None,
                 shuffle=True):
        self.image_folder = image_folder
        self.image_list_file = image_list_file
        self.transform = transform
        self.shuffle = shuffle
        self.data_list = self.read_list()

    def read_list(self):
        data_list = []
        with open(self.image_list_file) as f:
            for line in f:
                data_path = os.path.join(self.image_folder, line.split()[0])
                label_path = os.path.join(self.image_folder, line.split()[1])
                data_list.append((data_path, label_path))
        return data_list

    def preprocess(self, data, label):
        # h, w, c = data.shape
        # h_label, w_label = label.shape

        if self.transform:
            data, label = self.transform(data, label)

        label = label[:, :, np.newaxis]

        return data, label

    def __len__(self):
        return len(self.data_list)

    def __call__(self):
        for data_path, label_path in self.data_list:
            data = cv2.imread(data_path, cv2.IMREAD_COLOR)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)  # 转化为RGB格式
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            print(data.shape, label.shape)
            data, label = self.preprocess(data, label)
            yield data, label


class Transform(object):
    def __init__(self, size=256):
        self.size = size

    def __call__(self, data, label):
        data = cv2.resize(data, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        return data, label


def main():
    batch_size = 5
    place = fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        transform = Transform(256)

        # craete BasicDataloder instance

        image_folder = "day01/dummy_data"
        image_list_file = "day01/dummy_data/list.txt"
        basic_dataloader = BasicDataLoader(image_folder=image_folder,
                                           image_list_file=image_list_file,
                                           transform=transform,
                                           shuffle=True)

        # craete fluid.io.DataLoader instance
        dataloader = fluid.io.DataLoader.from_generator(capacity=1, use_multiprocess=False)

        # set sample generator for fluid dataloader
        dataloader.set_sample_generator(basic_dataloader,
                                        batch_size=batch_size,
                                        places=place)

        num_epoch = 2
        for epoch in range(1, num_epoch + 1):
            print(f'Epoch [{epoch}/{num_epoch}]:')
            for idx, (data, label) in enumerate(dataloader):
                print(f'Iter {idx}, Data shape: {data.shape}, Label shape: {label.shape}')


if __name__ == "__main__":
    main()
