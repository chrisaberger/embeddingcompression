import numpy as np
from statsmodels.multivariate import factor_rotation
class CF_Rotator(Rotator) :

    def __init__(self, objective):
        self.objective = objective

    def name(self):
        return "cf_"+objective

    def rotate(self, X):
        return factor_rotation.rotate_factors(X,self.objective)[0]
