import paddle
import paddle.fluid as fluid
import numpy as np
import cv2

eps = 1e-8

# 0 1 2 3
# 0 3 1 2
# 0 2 3 1

def Basic_SegLoss(preds, labels, ignore_index=255):
    n, c, h, w = preds.shape

    preds = fluid.layers.transpose(preds, [0, 2, 3, 1])
    # create softmax_with_cross_entropy criterion
    loss = fluid.layers.softmax_with_cross_entropy(logits=preds, label=labels, ignore_index=ignore_index)

    # transpose preds to NxHxWxC
    # preds = fluid.layers.transpose(preds, (n, h, w, c))
    mask = labels != ignore_index
    mask = fluid.layers.cast(mask, 'float32')

    # call criterion and compute loss
    loss = loss * mask
    avg_loss = fluid.layers.mean(loss) / (fluid.layers.mean(mask) + eps)

    return avg_loss


def main():
    label = cv2.imread('../day01/dummy_data/GroundTruth_trainval_png/2008_000002.png')
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY).astype(np.int64)
    pred = np.random.uniform(0, 1, (1, 59, label.shape[0], label.shape[1])).astype(np.float32)
    label = label[:, :, np.newaxis]
    label = label[np.newaxis, :, :, :]

    with fluid.dygraph.guard(fluid.CPUPlace()):
        pred = fluid.dygraph.to_variable(pred)
        label = fluid.dygraph.to_variable(label)
        loss = Basic_SegLoss(pred, label)
        print(loss)


if __name__ == "__main__":
    main()
