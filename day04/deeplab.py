import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Layer
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import BatchNorm
from paddle.fluid.dygraph import Dropout
from resnet_multi_grid import ResNet50


class ASPPPooling(Layer):
    def __init__(self, num_channels, num_filters):
        super(ASPPPooling, self).__init__()
        self.features = fluid.dygraph.Sequential(
            Conv2D(num_channels, num_filters, 1),
            BatchNorm(num_filters, act='relu')
        )

    def forward(self, inputs):
        n, c, h, w = inputs.shape

        x = fluid.layers.adaptive_pool2d(inputs, 1)
        x = self.features(x)
        x = fluid.layers.interpolate(x, (h, w), align_corners=False)

        return x


class ASPPConv(fluid.dygraph.Sequential):
    def __init__(self, num_channels, num_filters, dilation):
        super(ASPPConv, self).__init__(
            Conv2D(num_channels, num_filters, filter_size=3, padding=dilation, dilation=dilation),
            BatchNorm(num_filters, act='relu')
        )


class ASPPModule(Layer):
    def __init__(self, num_channels, num_filters, rates):
        super(ASPPModule, self).__init__()
        # concat
        self.features = []
        self.features.append(
            fluid.dygraph.Sequential(
                Conv2D(num_channels, num_filters, 1),
                BatchNorm(num_filters, act='relu')
            )
        )
        self.features.append(ASPPPooling(num_channels, num_filters))

        for r in rates:
            self.features.append(
                ASPPConv(num_channels, num_filters, r)
            )

        self.proj = fluid.dygraph.Sequential(
            Conv2D(num_filters * (2 + len(rates)), num_filters=256, filter_size=1, ),
            BatchNorm(256, act='relu'),
        )

    def forward(self, inputs):
        res = []
        for op in self.features:
            res.append(op(inputs))

        x = fluid.layers.concat(res, axis=1)
        x = self.proj(x)
        return x


class DeepLabHead(fluid.dygraph.Sequential):
    def __init__(self, num_channels, num_classes):
        super(DeepLabHead, self).__init__(
            ASPPModule(num_channels, 256, [12, 24, 36]),
            Conv2D(256, 256, 3, padding=1),  # 尺寸不变
            BatchNorm(256, act='relu'),
            Conv2D(256, num_classes, 1)  # 1*1conv
        )


class DeepLab(Layer):
    #
    def __init__(self, num_classes):
        super(DeepLab, self).__init__()
        res = ResNet50(pretrained=False)

        self.layer0 = fluid.dygraph.Sequential(
            res.conv,
            res.pool2d_max
        )
        self.layer1 = res.layer1
        self.layer2 = res.layer2
        self.layer3 = res.layer3
        self.layer4 = res.layer4

        #
        self.layer5 = res.layer5
        self.layer6 = res.layer6
        self.layer7 = res.layer7

        feature_dim = 2048
        self.classifier = DeepLabHead(feature_dim, num_classes)

    def forward(self, inputs):
        n, c, h, w = inputs.shape
        x = self.layer0(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        x = self.classifier(x)

        # 调整大小
        x = fluid.layers.interpolate(x, (h, w), align_corners=False)

        return x


def main():
    with fluid.dygraph.guard(fluid.CPUPlace()):
        x_data = np.random.rand(2, 3, 512, 512).astype(np.float32)
        x = to_variable(x_data)
        model = DeepLab(num_classes=59)
        model.eval()
        pred = model(x)
        print(pred.shape)

if __name__ == '__main__':
    main()
