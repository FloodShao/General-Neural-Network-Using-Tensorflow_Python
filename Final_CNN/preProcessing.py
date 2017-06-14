from scipy.io import loadmat as load
import matplotlib.pyplot as plt
import numpy as np


def reformat(samples, labels):
    # 改变原始数据的形状
    # 0	  1	  2	     3	   ->	3	    0	1	2
    #(高， 宽， 通道数， 图片数)->   (图片数， 高， 宽， 通道数)
    new = np.transpose(samples, (3, 0, 1, 2)).astype(np.float32)

    # labels change to one-hot encoding, [2] -> [0,0,1,0,0,0,0,0,0,0]
    # digit 0, represented as 10
    # lables change to one-hot encoding, [10] -> [1,0,0,0,0,0,0,0,0,0]

    labels = np.array([x[0] for x in labels])
    one_hot_labels = []
    for num in labels:
        one_hot = [0.0] * 10
        if num == 10:
            one_hot[0] = 1.0
        else:
            one_hot[num] = 1.0
        one_hot_labels.append(one_hot)

    labels = np.array(one_hot_labels).astype(np.float32)
    return new, labels


def normalize(samples):
    '''
    灰度化： 从三色通道 -> 单色通道
    (R+G+B)/3 -> 0-255
    再将图片从0-255 线性映射到-1.0 ~ 1.0
    '''
    a = np.add.reduce(samples, keepdims=True,
                      axis=3)  # right not axis=3 is the RGB value
    # keepdims, if set to be true, the axes which are reduced are left in the result as dimensions with size one
    # aixs, along which a reduction is performed
    a = a / 3.0
    return a / 128.0 - 1.0  # 线性映射


def distribution(labels, name):

    # 查看一下每个label的分布， 并画出统计图
    count = {}
    for label in labels:
        key = 0 if label[0] == 10 else label[0]
        if key in count:
            count[key] += 1
        else:
            count[key] = 1

    x = []
    y = []
    for k, v in count.items():
        x.append(k)
        y.append(v)

    y_pos = np.arange(len(x))
    plt.bar(y_pos, y, align='center', alpha=1)
    plt.xticks(y_pos, x)
    plt.ylabel('Count')
    plt.title(name + ' Distribution')
    plt.show()


def inspect(dataset, labels, i):
    # plt one pic
    print(labels[i])
    plt.imshow(dataset[i].squeeze())
    plt.show()
'''#the same as previous code
	if dataset.shape[3] == 1:
		shape = dataset.shape
		dataset = dataset.reshape(shape[0], shape[1], shape[2])
	plt.imshow(dataset[i])
'''


# 上层文件夹
traindata = load('../../data/train_32x32.mat')
testdata = load('../../data/test_32x32.mat')
# extradata = load('../data/extra_32x32.mat')

# print('Train Data Samples Shape:', traindata['X'].shape)
# print('Train Data Lables Shape:', traindata['y'].shape)

# print('Test Data Samples Shape:', testdata['X'].shape)
# print('Test Data Labels Shape:', testdata['y'].shape)

# print('Extra Data Samples Shape:', extradata['X'].shape)
# print('Extra Data Lables Shape:', extradata['y'].shape)

train_samples = traindata['X']
train_labels = traindata['y']

test_samples = testdata['X']
test_labels = testdata['y']

# extra_samples = extradata['X']
# extra_samples = extradata['y']
n_train_samples, _train_labels = reformat(train_samples, train_labels)
n_test_samples, _test_labels = reformat(test_samples, test_labels)

_train_samples = normalize(n_train_samples)
_test_samples = normalize(n_test_samples)

num_labels = 10
image_size = 32
num_channels = 1

if __name__ == '__main__':
    pass
    # inspect(_train_samples, _train_labels, 1)
    # _train_samples = normalize(_train_samples)
    # #inspect(_train_samples, _train_labels, 1)
    # distribution(train_labels, 'Train Labels')
    # #distribution(test_labels, 'Test Labels')
