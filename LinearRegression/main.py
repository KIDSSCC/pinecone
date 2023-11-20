import numpy as np
# 需要准备的环境不是paddle，而是paddlepaddle
import paddle


# 设置默认的全局dtype为float64
paddle.set_default_dtype("float64")
# 下载数据
print('下载并加载训练数据')
train_dataset = paddle.text.datasets.UCIHousing(mode='train')
eval_dataset = paddle.text.datasets.UCIHousing(mode='test')
train_loader = paddle.io.DataLoader(train_dataset, batch_size=32, shuffle=True)
eval_loader = paddle.io.DataLoader(eval_dataset, batch_size=8, shuffle=False)

print('加载完成')


# 定义全连接网络
class Regressor(paddle.nn.Layer):
    def __init__(self):
        super(Regressor, self).__init__()
        self.linear = paddle.nn.Linear(13, 1, None)

    def forward(self, input):
        return self.linear(input)
