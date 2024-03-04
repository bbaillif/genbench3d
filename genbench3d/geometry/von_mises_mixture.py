"""
Extracted from
https://framagit.org/fraschelle/mixture-of-von-mises-distributions/-/blob/master/vonMisesMixtures/mixture.py
"""

"""
Calculate and fits some periodic von Mises distribution functions. 

All the functions below can be called directly from the main package name. That is

```python
import vonMisesMixtures as vm

vm.density(x, mu, kapp)
vm.mixtures.density(x, mu, kappa)
```
return the same things, and the same is true for all the functions below.
"""

import numpy as np
import logging

from scipy.special import iv
from scipy.optimize import fsolve
from typing import Tuple 


class VonMisesMixture():
    '''
    To have the same API as GaussianMixture in sklearn
    n_components | an int | the number of von Mises distributions in the mixture
    threshold | a float | correspond to the euclidean distance between the old parameters and the new ones
    '''
    
    def __init__(self,
                 n_components: int = 10,
                 threshold: float = 1e-2) -> None:
        self.n_components = n_components
        self.threshold = threshold
        
    
    def fit(self,
            X: np.array) -> None:
        '''
        X | a 2D numpy array | represent the stochastic periodic process between -pi and pi
        '''
        
        assert len(X.shape) > 1, 'Input array X should be 2 dimensional (see sklearn)'
        logging.info('New fitting')
        
        X = X.reshape(-1)
        
        pi = np.random.random(self.n_components)
        mu = np.random.vonmises(0.0, 0.0, self.n_components)
        kappa = np.random.random(self.n_components)
        
        logging.info(f'Shape: {X.shape}')
        
        t = pi * self._density(X, mu, kappa)
        s = np.sum(t, axis=1)
        normalized_t = (t.T/s).T
        thresh = 1.0
        # calculate and update the coefficients, untill convergence
        while thresh > self.threshold:
            try:
                new_pi = np.mean(normalized_t, axis=0)
                new_mu = np.arctan2(np.sin(X) @ normalized_t, 
                                    np.cos(X) @ normalized_t)      
                c = np.cos(X) @ (normalized_t * np.cos(new_mu)) + np.sin(X) @ (normalized_t * np.sin(new_mu))
                k = lambda kappa: (c - iv(1, kappa) / iv(0, kappa) * np.sum(normalized_t, axis=0)).reshape(self.n_components)
                new_kappa = fsolve(k, np.zeros(self.n_components))
                
                # cap to 200 to allow high concentration but avoid overflow that happens
                # when kappa > 200
                new_kappa = np.where(new_kappa > 200, 200, new_kappa) 
                
                # kappa cannot be negative
                new_kappa = np.where(new_kappa < 1, 1, new_kappa)
                
                squared_pi_diff = (pi - new_pi) ** 2
                squared_mu_diff = (mu - new_mu) ** 2
                squared_log_kappa_diff = (np.log(kappa) - np.log(new_kappa)) ** 2
                thresh = np.sum(squared_pi_diff + squared_mu_diff + squared_log_kappa_diff)
                if np.isnan(thresh):
                    print('Nan thresh')
                    import pdb;pdb.set_trace()
                
                pi = new_pi
                mu = new_mu
                kappa = new_kappa
                
                t = pi * self._density(X, mu, kappa)
                s = np.sum(t, axis=1)
                normalized_t = (t.T / s).T
                
                # logging.info(f'Pi diff 2: {squared_pi_diff}')
                # logging.info(f'Mu diff 2: {squared_mu_diff}')
                # logging.info(f'Kappa diff 2: {squared_log_kappa_diff}')
                logging.info(f'Thresh: {thresh}')
                # logging.info(f'Pi: {pi}')
                # logging.info(f'Mu: {mu}')
                logging.info(f'Kappa: {kappa}')
                
            except Exception as e:
                print(e)
                import pdb;pdb.set_trace()
            
                
        self.means_ = mu.reshape(-1, 1) # to respect mean shape in GaussianMixture
        self.covariances_ = kappa.reshape(-1, 1, 1) # to respect covariance shape in GaussianMixture
        self.weights_ = pi
        # res = np.array([pi, mu, kappa])
        # in case there is no mixture, one fits the data using the estimators
        # if self.n_components == 1:
        #     res = self._pdfit(X)
        #     res = np.append(1.0, res)
        #     res = res.reshape(3, 1)
        # return res
    
    
    def score_samples(self,
                      X: np.array) -> float:
        assert len(X.shape) > 1, 'Input array X should be 2 dimensional (see sklearn)'
        X = X.reshape(-1)
        params = zip(self.means_.reshape(-1), self.covariances_.reshape(-1), self.weights_)
        values_list = []
        for mu, kappa, weight in params:
            values = self._density(X, mu, kappa)
            values = weight * values / np.sum(values)
            values_list.append(values)
        return np.sum(values_list, axis=0)


    def _density(self,
                 X: np.array, 
                mu: np.array, 
                kappa: np.array) -> np.array:
        """
        Calculate the von Mises density for a series x (a 1D numpy.array).
        
        Input | Type | Details
        -- | -- | --
        x | a 1D numpy.array of size L |
        mu | a 1D numpy.array of size n | the mean of the von Mises distributions
        kappa | a 1D numpy.array of size n | the dispersion of the von Mises distributions
        
        Output : 
            a (L x n) numpy array, L is the length of the series, and n is the size of the array containing the parameters. Each row of the output corresponds to a density
        """    
        not_normalized_density = np.array([np.exp(kappa * np.cos(i - mu)) for i in X])
        norm = 2 * np.pi * iv(0, kappa)
        density = not_normalized_density / norm
        return density


    # def _pdfit(self,
    #           series: np.array) -> Tuple[float]:
    #     """
    #     Calculate the estimator of the mean and deviation of a sample, for a von Mises distribution
        
    #     Input : 
    #         series : a 1D numpy.array
            
    #     Output : 
    #         the estimators of the parameters mu and kappa of a von Mises distribution, in a tuple (mu, kappa)
    #     See https://en.wikipedia.org/wiki/Von_Mises_distribution 
    #     for more details on the von Mises distribution and its parameters mu and kappa.
    #     """
    #     s0 = np.mean(np.sin(series))
    #     c0 = np.mean(np.cos(series))
    #     mu = np.arctan2(s0, c0)
    #     var = 1-np.sqrt(s0 ** 2 + c0 ** 2)
    #     k = lambda kappa: 1 - iv(1, kappa) / iv(0, kappa) - var
    #     kappa = fsolve(k, 0.0)[0]
    #     return mu, kappa 
