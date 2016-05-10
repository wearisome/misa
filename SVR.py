# -*- coding: utf-8 -*-
import numpy as np

class SVR:

    err = 0.0021
    eps = 0.001

    def __init__(self, data_y, data_x, rate):
        self.y = data_y
        self.x = data_x
        self.rate = rate

    def gaussian(self, x):
        tmp = []
        for x_n in x:
            x_n = np.exp(-x_n/2)
            tmp.append(x_n)
        return np.array(tmp)

    def kernel(self, x1, x2):
        x = -np.linalg.norm(x1-x2)
        return np.exp(x/2)

    def getPartGrad(self, alp, x, index):
        s = 0
        count = 0
        i = int(index/2) if(index%2==0) else int((index-1)/2)
        while count < len(self.y):
            x2 = self.x[count]
            s += (alp[2*count]-alp[2*count+1])*self.kernel(x, x2)
            count += 1
        result = self.y[i]-s/2-self.eps if(index%2==0) else -self.y[i]-self.eps+s/2
        return result
 
    def getGradient(self, alp):
        count = 0
        n_alp = []
        while count < len(self.y):
            x1 = self.x[count]
            n_alp.append(self.getPartGrad(alp, x1, 2*count))
            n_alp.append(self.getPartGrad(alp, x1, 2*count+1))
            count += 1
        return np.array(n_alp)        

    def getFunction(self, alp):
        count = 0
        palp = np.zeros(len(alp)/2)
        malp = np.zeros(len(alp)/2)
        while count < len(alp)/2:
            palp[count] = alp[2*count]
            malp[count] = alp[2*count+1]
            count += 1
        term1 = np.inner(palp-malp, self.y)
        term2 = self.eps*np.sum(palp+malp)
        term3 = 0
        count1 = 0
        count2 = 0
        while count1 < len(alp):
            x1 = self.x[count1]
            while count2 < len(alp):
                x2 = self.x[count2]
                term3 += (palp[count1]-malp[ccount1])*(palp[count2]-malp[ccount2])*self.kernel(x1, x2)
                count2 += 1
            count1 += 1
        return term1-term2-term3/2

    def hillClimbing(self):
        x = np.ones(2*len(self.y))
        norm_dx = 1
        while norm_dx > self.err:
            d_x = self.rate*self.getGradient(x)
            x  = x + d_x
            norm_dx = np.linalg.norm(d_x)
            print(d_x)
            print(norm_dx)
        return x

    def creatResult(self, alp, data):
        b = self.getPartGrad(alp, data, 0)
        y = 0
        count = 0
        while count < len(self.y):
            x2 = self.x[count]
            y += (alp[2*count]-alp[2*count+1])*self.kernel(data, x2)
            count += 1
        y += b
        return y

    def calcLoss(self, data_x, data_y):
        y = self.creatResult(self.result, data_x)
        loss = (y-data_y)*(y-data_y)
        return loss

    def update(self, data_x, data_y):
        x = []
        if self.calcLoss(data_x, data_y) > self.eps:
            d_x = self.rate*self.getGradient(self.result)
            x  = self.result + d_x
            self.resutl = np.array(x)
            #print(self.result)

    def executeLearn(self):
        alp1 = np.ones(2*len(self.y))
        alp1 = self.getGradient(alp1)
        self.result = self.hillClimbing()

    def getResult(self, data):
        return self.creatResult(self.result, data)


        
