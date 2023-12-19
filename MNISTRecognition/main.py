import struct, os
import numpy as np
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import matplotlib.pyplot as plt
import Classification


def load_mnist(image_file, label_file, path="mnist"):
    digits = np.arange(10)
    fname_image = os.path.join(path, image_file)
    fname_label = os.path.join(path, label_file)

    flbl = open(fname_label, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_image, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)

    images = zeros((N, rows * cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ind[i]*rows*cols : (ind[i]+1)*rows*cols]).reshape((1, rows*cols))
        labels[i] = lbl[ind[i]]
    return images, labels


def show_images(imgdata, imgtarget, show_column, show_row):
    plt.figure(figsize=(8, 6))
    for index, (im, it) in enumerate(list(zip(imgdata, imgtarget))):
        xx = im.reshape(28, 28)
        plt.subplots_adjust(left=0.1, bottom=None, right=0.9, top=None, wspace=None, hspace=None)
        plt.subplot(show_row, show_column, index+1)
        plt.axis('off')
        plt.imshow(xx, cmap='gray', interpolation='nearest')
        plt.title('label:%i' % it)
    plt.show()


if __name__ == '__main__':
    train_image, train_label = load_mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
    test_image, test_label = load_mnist("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
    # 进行数据的展示
    # show_images(train_image[:50], train_label[:50], 10, 5)
    # Max-Min归一化
    train_image = [im/255.0 for im in train_image]
    test_image = [im/255.0 for im in test_image]

    # 分类预测
    Classification.classification(train_image, train_label, test_image, test_label, 'SupportVectorMachine')





