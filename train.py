from PIL import Image
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from scipy.misc import imresize
import numpy as np
from feature import NPDFeature
from ensemble import AdaBoostClassifier
import os
import pickle
#from feature import NPDFeature
#from ensemble import AdaBoostClassifier

def get_data():
    List=[]
    facelist = os.listdir('./datasets/original/face/') # 
    for i in range(len(facelist)):
        location='./datasets/original/face/'+facelist[i]
        im = Image.open(location).convert('L') #读取图片，将全部图片转成大小为24*24的灰度图
        array = np.array(im).astype(float)
        array = imresize(array, (24,24))  # Convert between PIL image and NumPy ndarray
        #im2 = Image.fromarray(array)     # Convert between PIL image and NumPy ndarray
        npdfeature = NPDFeature(array)    #处理数据集数据，提取NPD特征。
        feature = npdfeature.extract()
        List.append(feature)

    nonfacelist = os.listdir('./datasets/original/nonface/') #
    for i in range(len(nonfacelist)):
        location='./datasets/original/nonface/'+nonfacelist[i]
        im = Image.open(location).convert('L') #读取图片，将全部图片转成大小为24*24的灰度图
        array = np.array(im).astype(float)
        array = imresize(array, (24,24))  # Convert between PIL image and NumPy ndarray
        #im2 = Image.fromarray(array)     # Convert between PIL image and NumPy ndarray
        npdfeature = NPDFeature(array)    #处理数据集数据，提取NPD特征。
        feature = npdfeature.extract()
        List.append(feature)

    file = open('feature.txt', 'wb')  # 用pickle库中的dump()函数将预处理后的特征数据保存到缓存中
    pickle.dump(List, file)
    file.close()


def load_data(X_trainsize,filename='feature.txt'): #            filename = 'feature.txt'
    file = open(filename, 'rb')  #使用load()函数读取特征数据
    List=pickle.load(file)
    file.close()
    X=np.array(List)   #    X.shape (1000,165600) 前500人脸，后500非人脸
    y = np.ones((1000,1))
    y[500:]=y[500:]*(-1)

    X_train=np.append(X[:int(X_trainsize/2),],X[500:(500+int(X_trainsize/2)),],axis=0)     #  训练集 X_train 和 y_train  (N,D)
    y_train=np.append(y[:int(X_trainsize/2),],y[500:(500+int(X_trainsize/2)),],axis=0)     #   将数据集切分为训练集和验证集    (N,1)
    X_test=np.append(X[int(X_trainsize/2):500,],X[(500+int(X_trainsize/2)):1000,],axis=0)
    y_test=np.append(y[int(X_trainsize/2):500,],y[(500+int(X_trainsize/2)):1000,],axis=0)
    return X_train,y_train,X_test,y_test

def Accuracy(y_predict,y_true):
    return np.mean(y_predict==y_true)

if __name__ == "__main__":
    # write your code here
    if os.path.exists("./feature.txt") is False:
        get_data()
        
    
    X_trainsize=800  #        训练集大小     (<1000)
    max_depth=3
    num_of_weakers=20
    Accuracytrainlist=[]
    Accuracytestlist=[]
    
    dataset=load_data(X_trainsize=X_trainsize)
    X_train,y_train,X_test,y_test=dataset[0],dataset[1],dataset[2],dataset[3]
    
    clf=DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=1)
    adaboost = AdaBoostClassifier(clf,n_weakers_limit=num_of_weakers)
    Accuracytrainlist,Accuracytestlist=adaboost.fit(X_train,y_train,X_test,y_test)
    print(Accuracy(adaboost.predict(X_test),y_test))
    print(classification_report(y_train, adaboost.predict(X_train),target_names=['non-face', 'face']))  #training_result
    print(classification_report(y_test, adaboost.predict(X_test),target_names=['non-face', 'face']))    #test_result

    plt.plot(Accuracytrainlist, 'black', label = 'Adaboost_train')
    plt.plot(Accuracytestlist, 'blue', label = 'Adaboost_test')
    plt.title('training Accuracy and test Accuracy')
    #plt.yscale('log')
    plt.xlabel('num_of_weakers')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

