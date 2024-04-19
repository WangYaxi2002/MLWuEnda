"""
线性回归
"""
import numpy as np


class LinerRegression:
    def __init__(self, x, y) -> None:
        self.__m = len(x)
        self.__x = np.array(x)
        self.__y = np.array(y)
        self.__w = 0
        self.__b = 0
        self.__history = []
        self.__cost = 0
        self.__update_cost()
        self.__update_history()

    def get_cost(self):
        return self.__cost

    def get_history(self):
        """
        获取梯度下降算法的历史纪录值
        """
        return self.__history

    def get_params(self):
        """
        获取线性回归得到的权重和偏置
        """
        return self.__w, self.__b

    def gradient_descent(self, lr, epoch):
        """
        使用梯度下降调整w和b
        """
        for i in range(epoch):
            self.__gradient_descent(lr)
            self.__update_cost()
            self.__update_history()
        return self.get_params()

    def __update_cost(self):
        self.__cost = (0.5/self.__m) * \
            sum((self.__w*self.__x + self.__b-self.__y) ** 2)

    def __update_history(self):
        self.__history.append([self.__w, self.__b, self.__cost])

    def __gradient_descent(self, lr):
        temp = self.__w
        self.__w = self.__w - lr * \
            sum((self.__w*self.__x+self.__b-self.__y)*self.__x)/self.__m
        self.__b = self.__b - lr * \
            sum(temp*self.__x+self.__b-self.__y)/self.__m

    def predict(self, x):
        """
        预测
        """
        return x*self.__w + self.__b
