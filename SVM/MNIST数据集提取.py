import numpy as np
import struct
import matplotlib.pyplot as plt

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

def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
     return decode_idx3_ubyte(idx_ubyte_file)

def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)

if __name__ == '__main__':
   
    test_images = load_test_images()
    test_labels = load_test_labels()

    # 查看前十个数据及其标签以读取是否正确
    for i in range(5):
        print(test_labels[i])
        plt.imshow(test_images[i], cmap='gray')
        plt.pause(0.1)
        plt.show()
    print('done')
