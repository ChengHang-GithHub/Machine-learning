import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from numpy import *
from scipy import * 
import numpy as np

a=2         #选择要加入的噪声类型

#定义椒盐函数
def SaltAndPepper(src,percetage): 
  SP_NoiseImg=src 
  SP_NoiseNum=int(percetage*src.shape[0]*src.shape[1]) 
  for i in range(SP_NoiseNum): 
    randX=random.random_integers(0,src.shape[0]-1) 
    randY=random.random_integers(0,src.shape[1]-1) 
    if random.random_integers(0,1)==0: 
      SP_NoiseImg[randX,randY]=0 
    else: 
      SP_NoiseImg[randX,randY]=15 
  return SP_NoiseImg

#定义高斯噪声
def addGaussianNoise(image,percetage): 
  G_Noiseimg = image 
  G_NoiseNum=int(percetage*image.shape[0]*image.shape[1]) 
  for i in range(G_NoiseNum): 
    temp_x = np.random.randint(1,8) 
    temp_y = np.random.randint(1,8) 
    G_Noiseimg[temp_x][temp_y] = 15 
  return G_Noiseimg

digits = datasets.load_digits()
SaltAndPepper_noiseImage=digits.images
G_noiseImage=digits.images
n_samples = len(digits.images)


if a==1:
  for i in range((n_samples // 2),(n_samples)):
    SaltAndPepper_noiseImage[i] = addGaussianNoise(digits.images[i],0.1)
else:
  for i in range((n_samples // 2),(n_samples)):
    G_noiseImage[i] = SaltAndPepper(digits.images[i],0.1)
    

data = digits.images.reshape((n_samples, -1))
classifier = svm.SVC(gamma=0.001)
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])
print("Classification report for classifier %s:\n%s\n"% (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected,predicted))

#加噪声后预测失误的部分预测样本输出(以图片形式)
count=0
recode=[]
for i in range(1,n_samples //2):
  if expected[i]!=predicted[i]:
     count=count+1
     recode.append(i+n_samples //2)
plt.figure(figsize=(13, 11), facecolor='r')#定义图片大小以及颜色
for i in range(0,count):
 if i >= 12:
     break
 plt.subplot(3, 4, i + 1)
 plt.imshow(digits.images[recode[i]],cmap=plt.cm.gray_r,interpolation='nearest')
 plt.title('wrong:%i,real:%i' % (expected[recode[i]-n_samples // 2], predicted[recode[i]-n_samples // 2]))
plt.show()
