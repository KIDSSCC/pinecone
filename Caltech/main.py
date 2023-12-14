
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import cv2
import paddle
import paddle.nn as nn
import numpy as np
import paddle.optimizer as optim

paddle.set_default_dtype("float32")
batch_size = 32


class CNN(paddle.nn.Layer):
    def __init__(self, num_class=102):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2D(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2D(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2D(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256*32*32, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512,num_class)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)



def prepare_test():
    train_path = 'dataset/train.txt'
    train_set = []

    batch_image = []
    batch_label = []
    with open(train_path, 'r') as file:
        for line in file:
            # line.strip()
            raw_picture, label = line.strip().split()
            raw_picture = 'dataset/images/' + raw_picture
            image = cv2.cvtColor(cv2.imread(raw_picture,), cv2.COLOR_BGR2RGB).astype(np.float32)
            image = np.resize(image, (256, 256, 3))
            image = np.transpose(image, (2, 0, 1))
            batch_image.append(image)
            batch_label.append(int(label))
            if len(batch_image) == batch_size:
                train_set.append([paddle.to_tensor(batch_image), paddle.unsqueeze(paddle.to_tensor(batch_label, dtype='int64'), axis=-1)])
                batch_image = []
                batch_label = []
            # batch_image.append(image)
    return train_set


if __name__ == '__main__':
    print('hello,world')
    train_set = prepare_test()
    print('get train_set')

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(learning_rate=0.001, parameters=model.parameters())
    print('begin train')
    num_epochs = 2
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for input, labels in train_set:
            optimizer.clear_grad()
            outputs = model(input)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            break
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_set)}')

    # 输出预测结果
    model.eval()
    test_path = 'dataset/test.txt'
    test_no_batch = []
    test_image_batch = []
    pred_res = []
    count = 0
    with open(test_path, 'r') as file:
        for line in file:
            count = count + 1
            pic_no = line.strip()
            test_no_batch.append(pic_no)
            pic_path = 'dataset/images/' + pic_no
            image = cv2.cvtColor(cv2.imread(pic_path), cv2.COLOR_BGR2RGB).astype(np.float32)
            image = np.resize(image, (256, 256, 3))
            image = np.transpose(image, (2, 0, 1))
            test_image_batch.append(image)
            if len(test_no_batch) == batch_size:
                outputs = model(paddle.to_tensor(test_image_batch))
                predicted_classes = paddle.argmax(outputs, axis=1)
                for index in range(batch_size):
                    pred_res.append([test_no_batch[index], predicted_classes[index]])
                count = 0
                test_no_batch = []
                test_image_batch = []
        if count != 0:
            outputs = model(paddle.to_tensor(test_image_batch))
            predicted_classes = paddle.argmax(outputs, axis=1)
            for index in range(len(test_no_batch)):
                pred_res.append([test_no_batch[index], predicted_classes[index]])
    print(len(pred_res))



