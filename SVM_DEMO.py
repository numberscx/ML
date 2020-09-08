import numpy as np
import math

class decSVM:
    def __init__(self,labels,max_iter=100, kernel='linear',gamma=20,C=1):
        self.max_iter = max_iter
        self._kernel = kernel
        self.gamma=gamma
        self.C=C
        self.model=[]
        self.labels=labels# 所有标签
        self.score=[]# 每个svm的效率，效率高的先决策
        self.classlabel=[]# 保存某个svm是分类那两个标签的
    def init_model(self,data,label):
        label_num=len(self.labels)
        for i in range(label_num):
            j=i+1
            while(j<label_num):
                moddata,modlabel=getdata(data,label,self.labels[i],self.labels[j])
                svmii=SVM(max_iter=self.max_iter,gamma=self.gamma,C=self.C)

                svmii.fit(moddata,modlabel)
                self.score.append(svmii.score(moddata,modlabel))
                print('svm %s,score= %s' %(len(self.score),self.score[-1]))
                self.model.append(svmii)
                # 分数越高，分类效果越好
                self.classlabel.append([self.labels[i],self.labels[j]])
                j+=1
    def predict(self,data):
        ans=[]
        for i in range(len(data)):
            ans.append(dec_proc(self.model,self.score,self.classlabel,data[i]))
        return ans

    def score1(self, X_test, y_test):
        ans = []
        for i in range(len(X_test)):
            #print("predict i",i)
            ans.append(dec_proc(self.model, self.score, self.classlabel, X_test[i]))
        right_count = 0
        for i in range(len(X_test)):
            if ans[i] == y_test[i][0]:
                right_count += 1
        return right_count / len(X_test)

def dec_proc(models,scores,classlabel,data,decision_function="grah"):
    if decision_function=="graph":
        if len(models)==1:
            ans=models[0].predict(data)
            if ans==1:
                return classlabel[0][0]
            else:
                return classlabel[0][1]
        else:
            seri=get_max_seri(scores)
            ans=models[seri].predict(data)
            # ans=1 | -1
            if ans==1:
                dellabel=classlabel[seri][0]
            else:
                dellabel=classlabel[seri][1]
            #dellabel=classlabel[seri].remove()
            delseri=[i for i in range(len(scores)) if(dellabel in classlabel[i])]
            #print("before models: %s,  scores: %s,  classlabel: %s  " %(models,scores,classlabel))
            models,scores,classlabel=delete_arr(delseri,models,scores,classlabel)
            #print("modfied models: %s,  scores: %s,  classlabel: %s  " % (models, scores, classlabel))
            return dec_proc(models,scores,classlabel,data)
    else:
        labelans=[]
        for i in range(len(models)):
            tempans=models[i].predict(data)
            # 每个模型判断一次，共判断n*(n-1)/2，并对所有判断得到的label数进行统计，得到最大值作为判段的类别
            if tempans==1:
                thislab=classlabel[i][0]
            else:
                thislab=classlabel[i][1]
            seri=[i for i in range(len(labelans)) if thislab in labelans[i]]
            if seri==[]:
                labelans.append([thislab,1])
            else:
                labelans[seri[0]][1]+=1
        tempmax=[i[1] for i in labelans]
        labelseri=tempmax.index(max(tempmax))
        return labelans[labelseri][0]

class SVM:
    def __init__(self, C=1,gamma=40,max_iter=500, kernel='RBF',ratio=1,ceof0=5):
        self.max_iter = max_iter
        self._kernel = kernel
        self.C=C
        self.gamma=gamma
        self.ratio=ratio
        self.C=[C,C*ratio]
        self.ceof0=ceof0
        self.lastalpha=[0,0,0]
        # 正样本和负样本的比值

    def init_args(self, features, labels):
        self.m, self.n = features.shape
        self.X = features
        self.Y = labels
        self.b = 0.0

        self.alpha = np.zeros(self.m)
        # 初始需要保持alpha*y=0的条件，因此初始化alpha[i]=0
        # m是样本数量
        # 将Ei保存在一个列表里
        self.E = [self._E(i) for i in range(self.m)]
        # 误差


    def _KKT(self, i):
        y_g = self._g(i) * self.Y[i]
        # 预测值和标签值相乘
        if self.alpha[i] == 0:
            return y_g >= 1
        elif 0 < self.alpha[i] < self.C[0] and self.Y[i]==1:
            return y_g == 1
        elif 0 < self.alpha[i] < self.C[1] and self.Y[i]==-1:
            return y_g == 1
        else:
            return y_g <= 1

    # g(x)预测值，输入xi（X[i]）
    def _g(self, i):
        # 初始时，所有样本的alpha都为1，也就是所有样本都是支持向量。
        r = 0
        r+=self.b
        for j in range(self.m):
            r += self.alpha[j] * self.Y[j] * self.kernel(self.X[i], self.X[j])
        return r
    # 核函数
    def kernel(self, x1, x2):
        if self._kernel == 'linear':
            return sum([x1[k] * x2[k] for k in range(self.n)])
        elif self._kernel == 'poly':
            return (self.gamma*sum([x1[k] * x2[k] for k in range(self.n)]) +self.ceof0)**3
        elif self._kernel == 'RBF':
            temp=[x1[k]-x2[k] for k in range(self.n)]
            temp=sum([temp[i]*temp[i] for i in range(self.n)])
            return pow(math.e,-temp/(2*self.gamma*self.gamma))
        elif self._kernel == 'sigmoid':
            return math.tanh(self.ceof0*sum([x1[k] * x2[k] for k in range(self.n)])+self.gamma)
        return 0

    # E（x）为g(x)对输入x的预测值和y的差
    def _E(self, i):
        return self._g(i) - self.Y[i]

    def _init_alpha(self):
        # 外层循环首先遍历所有满足0<a<C的样本点，检验是否满足KKT
        index_list = [i for i in range(self.m) if (0 < self.alpha[i] < self.C[0] and self.Y[i]==1) or (0 < self.alpha[i] < self.C[1] and self.Y[i]==-1)]
        # 否则遍历整个训练集
        non_satisfy_list = [i for i in range(self.m) if i not in index_list]
        index_list.extend(non_satisfy_list)
        for i in index_list:
            if self._KKT(i):
                continue
            #print("KKT choose =",i)
            E1 = self.E[i]
            # 如果E1是+，选择最小的；如果E1是负的，选择最大的
            if E1 >= 0:
                j = min(range(self.m), key=lambda x: self.E[x])
            else:
                j = max(range(self.m), key=lambda x: self.E[x])
            if i==self.lastalpha[0] and j==self.lastalpha[1] or i==self.lastalpha[0] and j==self.lastalpha[1]:
                self.lastalpha[2]+=1
            else:
                self.lastalpha[0]=i
                self.lastalpha[1]=j
                self.lastalpha[2]=0
            #print("kkt choose",i,j)
            return i, j
        return 0,0

    def _compare(self, _alpha, L, H):
        if _alpha > H:
            return H
        elif _alpha < L:
            return L
        else:
            return _alpha

    def fit(self, features, labels):
        self.init_args(features, labels)
        # 初始化样本数，特征数目
        for t in range(self.max_iter):
            # train
            i1, i2 = self._init_alpha()
            if self.lastalpha[2]>=3:
                break
            if (i1+i2==0):
                break
            # 边界
            if self.Y[i1] == self.Y[i2]:
                Lz = max(0, self.alpha[i1] + self.alpha[i2] - self.C[0])
                Hz = min(self.C[0], self.alpha[i1] + self.alpha[i2])
                Lf = max(0, self.alpha[i1] + self.alpha[i2] - self.C[1])
                Hf = min(self.C[1], self.alpha[i1] + self.alpha[i2])
            else:
                Lz = max(0, self.alpha[i2] - self.alpha[i1])
                Hz = min(self.C[0], self.C[0] + self.alpha[i2] - self.alpha[i1])
                Lf = max(0, self.alpha[i2] - self.alpha[i1])
                Hf = min(self.C[1], self.C[1] + self.alpha[i2] - self.alpha[i1])
            E1 = self.E[i1]
            E2 = self.E[i2]
            # eta=K11+K22-2K12
            eta = self.kernel(self.X[i1], self.X[i1]) + self.kernel(
                self.X[i2],
                self.X[i2]) - 2 * self.kernel(self.X[i1], self.X[i2])
            if eta <= 0:
                print('eta <= 0')
                continue

            alpha2_new_unc = self.alpha[i2] + self.Y[i2] * (
                E1 - E2) / eta  #此处有修改，根据书上应该是E1 - E2，书上130-131页

            if self.Y[i2]==1:
                alpha2_new = self._compare(alpha2_new_unc, Lz, Hz)
            else:
                alpha2_new = self._compare(alpha2_new_unc, Lf, Hf)
            alpha1_new = self.alpha[i1] + self.Y[i1] * self.Y[i2] * (
                self.alpha[i2] - alpha2_new)

            b1_new = -E1 - self.Y[i1] * self.kernel(self.X[i1], self.X[i1]) * (
                alpha1_new - self.alpha[i1]) - self.Y[i2] * self.kernel(
                    self.X[i2],
                    self.X[i1]) * (alpha2_new - self.alpha[i2]) + self.b
            b2_new = -E2 - self.Y[i1] * self.kernel(self.X[i1], self.X[i2]) * (
                alpha1_new - self.alpha[i1]) - self.Y[i2] * self.kernel(
                    self.X[i2],
                    self.X[i2]) * (alpha2_new - self.alpha[i2]) + self.b
            #print("epoch %s,b1new %s,b2new %s" %(t,b1_new,b2_new))
            if (0 < alpha1_new < self.C[0] and self.Y[i1]==1) or (0 < alpha1_new < self.C[1] and self.Y[i1]==-1):
                b_new = b1_new
            elif (0 < alpha2_new < self.C[0] and self.Y[i2]==1) or (0 < alpha2_new < self.C[1] and self.Y[i2]==-1):
                b_new = b2_new
            else:
                # 选择中点
                b_new = (b1_new + b2_new) / 2

            # 更新参数
            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new
            #print("epoch %s,error from %s->%s,%s->%s" %(t,self.E[i1],self._E(i1),self.E[i2],self._E(i2)))
            self.E[i1] = self._E(i1)
            self.E[i2] = self._E(i2)
        print(self.alpha)
        return 'train done!'

    def predict(self, data):
        r = 0
        r+=self.b
        # 直接让r=self.b，r变化，则b也变化
        for i in range(self.m):
            r += self.alpha[i] * self.Y[i] * self.kernel(data, self.X[i])
        if(r>0):
            return 1
        else:
            return -1

    def score(self, X_test, y_test):
        right_count = 0
        for i in range(len(X_test)):
            result = self.predict(X_test[i])
            if result == y_test[i]:
                right_count += 1
        return right_count / len(X_test)

    def _weight(self):
        # linear model
        yx = self.Y.reshape(-1, 1) * self.X
        self.w = np.dot(yx.T, self.alpha)
        return self.w

def delete_arr(seris,a,b,c):
    seris.sort(reverse=True)
    for i in seris:
        a.pop(i)
        b.pop(i)
        c.pop(i)
    return a,b,c
def getdata(data,label,la,lb):
    '''
    :param data:
    :param label:
    :param la: first label name，标签改为1
    :param lb: next label name，标签改为-1
    :return:
    '''
    data_num=np.shape(data)[0]
    seria=[i for i in range(data_num) if label[i]==la]
    serib=[i for i in range(data_num) if label[i]==lb]
    data=np.array(data)
    #label=np.array(label)
    moddata=np.vstack((data[seria],data[serib]))
    modlabel=np.ones((np.shape(moddata)[0],1))
    modlabel[-len(serib):,0]=-1
    #modlabel=np.vstack((label[seria],label[serib]))
    return moddata,modlabel
def get_max_seri(arr):
    max=arr[0]
    seri=0
    i=1
    while(i<len(arr)):
        if arr[i]>max:
            max=arr[i]
            seri=1
        i+=1
    return seri
