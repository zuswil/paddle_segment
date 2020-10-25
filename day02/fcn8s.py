import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import Conv2DTranspose
from paddle.fluid.dygraph import Dropout
from paddle.fluid.dygraph import BatchNorm
from paddle.fluid.dygraph import Pool2D
from paddle.fluid.dygraph import Linear
from vgg import VGG16BN


class FCN8s(fluid.dygraph.Layer):
    # create fcn8s model
    def __init__(self, num_classes=59):
        super(FCN8s, self).__init__()
        backbone = VGG16BN(pretrained=False)
        self.numclasses = num_classes
        self.layer1 = backbone.layer1
        self.layer1[0].conv._padding = [100, 100]
        self.pool1 = Pool2D(pool_size=2, pool_stride=2, ceil_mode=True)
        self.layer2 = backbone.layer2
        self.pool2 = Pool2D(pool_size=2, pool_stride=2, ceil_mode=True)
        self.layer3 = backbone.layer3
        self.pool3 = Pool2D(pool_size=2, pool_stride=2, ceil_mode=True)
        self.layer4 = backbone.layer4
        self.pool4 = Pool2D(pool_size=2, pool_stride=2, ceil_mode=True)
        self.layer5 = backbone.layer5
        self.pool5 = Pool2D(pool_size=2, pool_stride=2, ceil_mode=True)

        self.fc6 = Conv2D(512, 4096, 7, act='relu')
        self.fc7 = Conv2D(4096, 4096, 1, act='relu')

        self.drop6 = Dropout()
        self.drop7 = Dropout()

        self.score1 = Conv2D(4096, num_classes, 1)
        self.score2 = Conv2D(256, num_classes, 1)
        self.score3 = Conv2D(512, num_classes, 1)

        self.upsampling = Conv2DTranspose(num_channels=num_classes,
                                          num_filters=num_classes,
                                          filter_size=4,
                                          stride=2,
                                          bias_attr=False)
        self.upsampling2 = Conv2DTranspose(num_channels=num_classes,
                                           num_filters=num_classes,
                                           filter_size=4,
                                           stride=2,
                                           bias_attr=False)
        self.upsampling_final = Conv2DTranspose(num_channels=num_classes,
                                                num_filters=num_classes,
                                                filter_size=16,
                                                stride=8,
                                                bias_attr=False)

    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.pool1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = self.layer3(x)
        x = self.pool3(x)
        pool3 = x
        x = self.layer4(x)
        x = self.pool4(x)
        pool4 = x
        x = self.layer5(x)
        x = self.pool5(x)

        x = self.fc6(x)
        x = self.drop6(x)
        x = self.fc7(x)
        x = self.drop7(x)

        x = self.score1(x)
        x = self.upsampling(x)
        # 一次上采样
        up_output = x  # 1/16
        x = self.score3(pool4)
        x = x[:, :, 5:5 + up_output.shape[2], 5:5 + up_output.shape[3]]

        # crop and copy
        up_pool4 = x
        x = up_pool4 + up_output
        x = self.upsampling2(x)
        up_pool4 = x
        x = self.score2(pool3)
        x = x[:, :, 9:9 + up_pool4.shape[2], 9:9 + up_pool4.shape[3]]
        up_pool3 = x  # 1/8

        x = up_pool3 + up_pool4
        x = self.upsampling_final(x)
        x = x[:, :, 31:31 + inputs.shape[2], 31:31 + inputs.shape[3]]
        return x


def main():
    place = fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        x_data = np.random.rand(2, 3, 512, 512).astype(np.float32)
        x = to_variable(x_data)
        model = FCN8s(num_classes=59)
        model.eval()
        pred = model(x)
        print(pred.shape)


if __name__ == '__main__':
    main()
