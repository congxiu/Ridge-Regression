# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 17:59:41 2015

@author: congxiu
"""
import numpy as np

class CrossValidation:
    def __init__(self, sample, label, model, k = 10, log10Params = np.arange(2, -11, -1)):
        self.sample = sample
        self.label = label
        self.size = sample.shape[0]
        self.model = model
        self.k = k
        self.log10Params = log10Params
        self.bestParam = None
        
    def train(self):
        shuffledIDX = np.random.permutation(range(self.size))
        self.sample = self.sample[shuffledIDX]
        self.label = self.label[shuffledIDX]
        bucketSize = self.size / self.k
        buckets = range(0, self.size, bucketSize) + [self.size]
        errors = []
        for log10Param in self.log10Params:
            CVerr = 0
            for idx, dataIdx in enumerate(buckets[:-1]):
                trainSample = np.concatenate([self.sample[:dataIdx],
                                              self.sample[buckets[idx + 1]:]])
                testSample = self.sample[dataIdx:buckets[idx + 1]]
                trainLabel = np.concatenate([self.label[:dataIdx],
                                              self.label[buckets[idx + 1]:]])
                testLabel = self.label[dataIdx:buckets[idx + 1]]
                
                currModel = self.model(trainSample, trainLabel)
                currModel.train(10 ** log10Param)
                CVerr += 1 - currModel.score(testSample, testLabel, True)
            
            CVerr = CVerr / float(self.k)
            errors.append(CVerr)
            
        self.bestParam = self.log10Params[np.argmin(errors)]
        return errors   
                

class RidgeRegression:
    def __init__(self, sample, label):
        self.sample = sample
        self.label = label
        self.size = sample.shape[0]
        self.w = np.array([0] * sample.shape[1], ndmin = 2).T
        
    def reset(self, sample, label):
        self.sample = sample
        self.label = label
        self.size = sample.shape[0]
        self.w = np.array([0] * sample.shape[1], ndmin = 2).T
        
    def train(self, Lambda = 0.001, analytic = True):
        if analytic:
            hatM = np.linalg.inv((self.sample.T.dot(self.sample) + Lambda * np.identity(len(self.w)))).dot(self.sample.T)
            self.w = hatM.dot(self.label)
            
            print "Training score is", self.score(self.sample, self.label)
            return 
            
    def predict(self, data):
        return data.dot(self.w)
        
    def score(self, sample, label, classification = False):
        if classification:
            return (np.sign(self.predict(sample)) == np.sign(label)).mean()
            
        return (abs(self.predict(sample) - label) ** 2).sum() / float(len(label))