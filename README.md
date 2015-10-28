# Ridge-Regression
A simple Ridge Regression model with Cross Validation

CV = CrossValidation(sample, label, RidgeRegression)
CV.train()

RR = RidgeRegression(sample, label)

RR.train(Lambda = CV.bestParam)

