# -*- coding: utf-8 -*-
import numpy as np

class SVR:

    err = 0.001
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

    def calcAbsSum(self, alp):
        s = 0
        for a in alp:
            s += abs(a)
        return(s)

    def getPartGrad(self, alp, x, index):
        s = 0
        count = 0
        while count < len(alp):
            x2 = self.x[count]
            s += alp[count]*self.kernel(x, x2)
            count += 1
        return self.y[index]-self.eps-s/2
 
    def getGradient(self, alp):
        count = 0
        n_alp = []
        while count < len(self.y):
            x1 = self.x[count]
            n_alp.append(self.getPartGrad(alp, x1, count))
            count += 1
        return np.array(n_alp)

    def getFunction(self, alp):
        term1 = np.inner(alp, self.y)
        term2 = self.eps*self.calcAbsSum(alp)
        term3 = 0
        count1 = 0
        count2 = 0
        while count1 < len(alp):
            x1 = self.x[count1]
            while count2 < len(alp):
                x2 = self.x[count2]
                term3 += alp[count1]*alp[count2]*self.kernel(x1, x2)
                count2 += 1
            count1 += 1
        return term1-term2-term3/2

    def hillClimbing(self):
        x = np.ones(len(self.y))
        norm_dx = 1
        while norm_dx > self.err:
            d_x = self.rate*self.getGradient(x)
            x  = x + d_x
            norm_dx = np.linalg.norm(d_x)
            #print(d_x)
            #print(norm_dx)
        return x

    def creatResult(self, alp, data):
        b = self.getPartGrad(alp, data, 0)
        y = 0
        count = 0
        while count < len(alp):
            x2 = self.x[count]
            y += alp[count]*self.kernel(data, x2)
            count += 1
        y += b
        return y 

    def executeLearn(self):
        alp1 = np.ones(len(self.y))
        alp1 = self.getGradient(alp1)
        self.result = self.hillClimbing()

    def getResult(self, data):
        return self.creatResult(self.result, data)


        
