import numpy as np

class SphereDistribution():


    def __init__(self,n_sphere):

        self.n_sphere = n_sphere


    def __call__(self, radius:float):

        def wrapper(size = 1):

            sample = np.random.uniform(-radius,radius,(self.n_sphere,size))
            return sample / (np.sum(sample**2))**.5

        return wrapper