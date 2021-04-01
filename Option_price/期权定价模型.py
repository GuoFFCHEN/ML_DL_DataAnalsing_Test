# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 18:58:06 2021

@author: chen_zhihan
"""

# ###############################################
# #期权的定价

import numpy as np
import math

E = math.e

class Binomial_tree_sim:
    def __init__(self, r, sigma, S_0, K, T, steps, option_type="european", call_or_put="call"):

        self.r = r
        self.sigma = sigma
        self.S_0 = S_0
        self.K = K
        self.T = T
        self.steps = steps
        
        self.option_type = option_type
        self.call_or_put = call_or_put
        
        # 计算出树形分叉参数
        self.dt = self.T/self.steps
        self.u = E**(self.sigma*self.dt**0.5)
        self.d = 1/self.u
        self.p = (E**(self.r*self.dt)-self.d)/(self.u-self.d)
        
        # 将会得到的结果
        self.tree = None
        self.option_price = None
        
        # 计算出一个树形
        self.build_tree()
        
    def build_tree(self):
        """ 计算出股票价格在树形上每个节点的价格。
        """
        self.tree = list()
        for lvl in range(self.steps+1):
            row = list()
            for j in range(lvl+1):
                node = dict()
                node["stock_price"] = self.S_0*self.u**(j)*self.d**(lvl-j)
                node["option_price"] = None
                row.append(node)
            self.tree.append(row)
        return
    
    def calculate_option_price(self):
        """ 计算给定类型期权的价格。
        """
        # 简化参数名称。
        r, K, steps = self.r, self.K, self.steps
        dt, p = self.dt, self.p
        
        # 计算出期权在树形末端的价格。
        for node in self.tree[-1]:
            # 如果是看涨期权。
            if self.call_or_put == "call":
                node["option_price"] = max(node["stock_price"]-K, 0)
            # 如果是看跌期权。
            else:
                node["option_price"] = max(K-node["stock_price"], 0)
        
        # 如果是欧式期权。
        if self.option_type == "european":
            # 递推出树形根节点期权的价格。
            for lvl in range(steps-1, -1, -1):
                for j in range(len(self.tree[lvl])):
                    self.tree[lvl][j]["option_price"] = E**(-r*dt)*(p*self.tree[lvl+1][j+1]["option_price"]+\
                                                    (1-p)*self.tree[lvl+1][j]["option_price"])
        
        # 如果是美式期权，过程同欧式期权，计算节点价格时考虑需不需要在该节点执行。
        else:
            for lvl in range(self.steps-1, -1, -1):
                for j in range(len(self.tree[lvl])):
                    self.tree[lvl][j]["option_price"] = E**(-r*dt)*(p*self.tree[lvl+1][j+1]["option_price"]+\
                                                    (1-p)*self.tree[lvl+1][j]["option_price"])
                    # 考虑要不要这时执行。
                    if self.call_or_put == "call":
                        self.tree[lvl][j]["option_price"] = max(self.tree[lvl][j]["option_price"], \
                                                            self.tree[lvl][j]["stock_price"]-K)
                    else:
                        self.tree[lvl][j]["option_price"] = max(self.tree[lvl][j]["option_price"], \
                                                            K-self.tree[lvl][j]["stock_price"])
        
        self.option_price = self.tree[0][0]["option_price"]

        return 

option = Binomial_tree_sim(0.05, 0.2, 10, 10, 3, 10)
option.calculate_option_price()
print(option.option_price)
tree = option.tree
