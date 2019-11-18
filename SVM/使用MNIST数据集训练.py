import numpy as np
import struct
import matplotlib.pyplot as plt
from sklearn import svm,metrics      #导入sklearn的SVM模块，评价指标

# 训练集文件
train_images_idx3_ubyte_file = 'C:/Users/成行/Desktop/mnist数据集/train-images-idx3-ubyte/train-images.idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = 'C:/Users/成行/Desktop/mnist数据集/train-labels-idx1-ubyte/train-labels.idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file='C:/Users/成行/Desktop/mnist数据集/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file='C:/Users/成行/Desktop/mnist数据集/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte'

def decode_idx3_ubyte(idx3_ubyte_file):
    
     # 读取二进制数据
     bin_data = open(idx3_ubyte_file, 'rb').read()

     
     # 解析文件头信息
     offset = 0
     fmt_header = '>iiii' #因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
     #该函数可以将缓冲区bin_data中的内容按照指定的格式fmt='someformat'读取，从偏移量为offset=numb的位置开始进行读取。返回的是一个对应的元组tuple，一般使用的场景是从一个二进制或者其他文件中读取的内容进行解析操作。
     magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
     print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

     # 解析数据集
     image_size = num_rows * num_cols
     offset += struct.calcsize(fmt_header)  #获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。
     print(offset)
     fmt_image = '>' + str(image_size) + 'B'  #图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
     print(fmt_image,offset,struct.calcsize(fmt_image))
     images = np.empty((num_images, num_rows, num_cols))
     for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape(num_rows, num_cols)
        offset += struct.calcsize(fmt_image)
     return images

def decode_idx1_ubyte(idx1_ubyte_file):
     # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()
    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))
    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print ('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels
def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
     return decode_idx3_ubyte(idx_ubyte_file)
def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)
def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
     return decode_idx3_ubyte(idx_ubyte_file)
def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)

if __name__ == '__main__':

    train_images = load_train_images()
    train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()
    images_and_labels = list(zip(train_images, train_labels))#将用作训练的样本的 image 属性和 target 属性合并到一个列表中
    print(images_and_labels[:1])
    print(train_labels[:10])
    print('the sum of samples=%i\n'%test_labels.size)
    #第 5—9 行代码用以排布最终结果的总体界面，以两行四列的形式展示，其中第 8—9 行代码用以展示训练样本中前四个数据的图片和其对应的标签。
    for index, (image, label) in enumerate(images_and_labels[:10]):#对于列表的前四项
      plt.subplot(4, 5, index + 1)#定义子图的排布方式
      plt.axis('off')#关闭坐标轴
      plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')#定义图片展示方式：灰色图片，插值方式为:nearest
      plt.title('Training: %i' % label)#为每张图片绘制标题
    n_samples = len(train_labels)#得到训练样本总数量
    data = train_images.reshape((n_samples, -1))#将原本 28*28 的矩阵转化为 784 维的特征向量
    classifier = svm.SVC(gamma=0.001)  #设计svm多分类器参数（此时设置为最优参数）
    classifier.fit(data[:n_samples//6],train_labels[:n_samples//6])#利用MNIST数据集对模型进行训练
    n_samples1 = len(test_labels)
    expected = test_labels       #将样本测试集的标签赋值给数组 expected
    data1 = test_images.reshape((n_samples1, -1))
    predicted = classifier.predict(data1)#利用之前训练过的模型对训练数据集进行预测，赋值给数组 predicted
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))#利用自带函数生成预测精度矩阵
    images_and_predictions = list(zip(test_images, predicted))
    for index, (image, predicted) in enumerate(images_and_predictions[:10]):
      plt.subplot(4, 5, index + 11)
      plt.axis('off')
      plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
      plt.title('Prediction: %i' % predicted)
    plt.show()
