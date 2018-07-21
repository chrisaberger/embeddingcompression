import numpy as np
from .rotator import Rotator
from statsmodels.multivariate import factor_rotation
class CF_Rotator(Rotator) :

    def __init__(self, objective):
        self.objective = objective

    def name(self):
        return "cf_"+objective

    def rotate(self, X):
        print("Applying a "+self.objective+" rotation to the embeddings matrix ...")
        return factor_rotation.rotate_factors(X,self.objective)[0]
