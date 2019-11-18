import matplotlib.pyplot as plt                 #导入matplotlib库
from sklearn import datasets, svm, metrics      #导入sklearn的数据集，SVM模块，评价指标
digits = datasets.load_digits()#将库中已有的手写数字数据集（digits）赋值给变量 digits
images_and_labels = list(zip(digits.images, digits.target))#将用作训练的样本的 image 属性和 target 属性合并到一个列表中
print(images_and_labels[0])
print('the sum of samples=%i\n'%digits.target.size)
#第 5—9 行代码用以排布最终结果的总体界面，以两行四列的形式展示，其中第 8—9 行代码用以展示训练样本中前四个数据的图片和其对应的标签。
for index, (image, label) in enumerate(images_and_labels[:10]):#对于列表的前四项
   plt.subplot(4, 5, index + 1)#定义子图的排布方式
   plt.axis('off')#关闭坐标轴
   plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')#定义图片展示方式：灰色图片，插值方式为:nearest
   plt.title('Training: %i' % label)#为每张图片绘制标题
n_samples = len(digits.images)#得到样本总数量
data = digits.images.reshape((n_samples, -1))#将原本 8*8 的矩阵转化为 64 维的特征向量
classifier = svm.SVC(gamma=0.002,C=2.6)  #设计svm多分类器参数（此时设置为最优参数）
print(digits.target[:10])
print(data[:1])
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])#对样本中的前一半的数据对模型进行训练
expected = digits.target[n_samples // 2:]       #将样本中后一半的标签赋值给数组 expected
predicted = classifier.predict(data[n_samples // 2:])#利用之前训练过的模型对样本中后一半的数据进行预测，赋值给数组 predicted
print("Classification report for classifier %s:\n\n%s\n"% (classifier, metrics.classification_report(expected, predicted)))#利用库中自带的函数生成分类器的分类报告，构建显示主要分类指标的文本报告
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))#利用自带函数生成预测精度矩阵
images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:10]):
   plt.subplot(4, 5, index + 11)
   plt.axis('off')
   plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
   plt.title('Prediction: %i' % prediction)
plt.show()
