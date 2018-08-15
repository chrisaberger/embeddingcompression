import numpy as np
from .rotator import Rotator
from statsmodels.multivariate import factor_rotation
class CF_Rotator(Rotator) :

    def __init__(self, objective):
        self.objective = objective

    def name(self):
        return "cf_"+objective

    def rotate(self, X):
        print("Applying the "+self.objective+" rotation to the matrix ...")
        print(self.objective[-3:-1])
        if self.objective[-3:] == "max" or self.objective[0:3] == "par":
            #for this class of rotations, we can use the analytic solution
            print("gpa")
            return factor_rotation.rotate_factors(X,self.objective,'gpa',max_tries=250)[0]
        else: 
            return factor_rotation.rotate_factors(X,self.objective,'analytic')[0]
