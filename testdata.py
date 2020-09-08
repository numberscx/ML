from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
import SVM_DEMO as dSVM
import numpy as np
import matplotlib.pyplot as plt
import test as test
from sklearn import svm
import pso_test as pso
from sklearn.datasets import load_iris,load_wine,load_breast_cancer

def plot_scatter(x,y,label):
    seriz=[i for i in range(len(x)) if label[i]==1]
    serif = [i for i in range(len(x)) if label[i] != 1]
    plt.scatter(x[seriz], y[seriz],color='red',s=3)
    plt.scatter(x[serif], y[serif], color='blue',s=3)
    plt.show()

def get_guass():
    #2
    x1, y1 = make_gaussian_quantiles(n_samples=1000, n_features=2, n_classes=2, mean=[1, 2], cov=2)
    #plot_scatter(x1[:,0],x1[:,1],y1)
    return x1,y1

def get_kmean():
    #1
    x1, y1 = make_blobs(n_samples=1000, n_features=2, centers=[[-1, 0], [0, -1]], cluster_std=[0.4, 0.5])
    #plot_scatter(x1[:, 0], x1[:, 1], y1)
    return x1, y1

def get_poly():
    #3
    x1, y1 = make_blobs(n_samples=1000, n_features=2, centers=[[0, 0], [1, 1], [2, 2]], cluster_std=[0.4, 0.5, 0.2])
    seri=[i for i in range(len(x1)) if y1[i]==2]
    y1[seri]=0
    #plot_scatter(x1[:, 0], x1[:, 1], y1)
    return x1, y1

def diff_kernel():
    x,y=get_poly()
    seri=[i for i in range(len(x)) if y[i]==0]
    y[seri]=-1
    label=np.expand_dims(y,axis=1)
    #print(np.shape(label))
    print("begin pso")
    #C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大
    pre_fun = dSVM.SVM(C=0.1, gamma=0.5, kernel='poly',ceof0=5,max_iter=100)
    #pre_fun=svm.SVC(C=0.1, kernel='poly', decision_function_shape='ovo',coef0=5)
    pre_fun.fit(x, y.ravel())
    #print(pre_fun.score(x, y))
    print('begin fit')
    #pre_fun.fit(x, y)
    # print(pre_fun.alpha)
    print("预测分数", pre_fun.score(x, y))
    print('fit over ,begin chart')
    test.plot_decision_boundary(pre_fun, x, label)

def poly_fit(para):
    x,y=get_poly()
    seri=[i for i in range(len(x)) if y[i]==0]
    y[seri]=-1
    label=np.expand_dims(y,axis=1)
    pre_fun = dSVM.SVM(C=1, gamma=para[0], kernel='poly',pod=para[1], max_iter=20)
    # pre_fun = dSVM.SVM(C=1, gamma=10, kernel='poly',max_iter=100)
    #print('begin fit')
    pre_fun.fit(x, y)
    # print(pre_fun.alpha)
    fitn=pre_fun.score(x, y)
    print("para",para)
    print("预测分数", fitn)
    return fitn

def demosvm_vs_sklearn():
    data=load_breast_cancer()
    x=data['data']
    label=data['target']

    demosvm = dSVM.decSVM(labels=[0,1], kernel='RBF', max_iter=100,gamma=0.1,C=10)
    sksvm=svm.SVC(C=1, decision_function_shape='ovo',coef0=5)
    demosvm.init_model(x,label)
    sksvm.fit(x,label)
    testy=np.expand_dims(label,axis=1)
    print("demo",demosvm.score1(x,testy))
    print("sksvm",sksvm.score(x,label))
def find_gamma():
    i=0.5
    ans=[]
    data = load_breast_cancer()
    x = data['data']
    label = data['target']
    testy = np.expand_dims(label, axis=1)
    while(i<100):
        demosvm = dSVM.decSVM(labels=[0, 1], kernel='RBF', max_iter=100, gamma=i)
        #sksvm = svm.SVC(C=1, decision_function_shape='ovo', coef0=5)
        demosvm.init_model(x, label)
        #sksvm.fit(x, label)
        ans.append(demosvm.score1(x, testy))
        print("gamma= %s,score= %s" %(i,ans[-1]))
        i+=0.5
    print(ans)
    plt.plot(ans)
    plt.show()
#get_guass()
#get_kmean()
#get_poly()
#diff_kernel()
demosvm_vs_sklearn()
#find_gamma()