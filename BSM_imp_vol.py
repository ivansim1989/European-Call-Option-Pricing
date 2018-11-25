import numpy as np
from scipy import stats
from scipy.optimize import fsolve
from scipy.stats import norm

class call_option(object):
    '''Class for Eupopean call options in BSM Model.
    
    Attributes
    ==============
    S0: float initial stock/index level
    K: float strike price
    t: dataime/Time stamp object pricing date
    M: date/Timestamp object materity date
    r: float contant risk-free short rate
    sigma : float volatility factor in diffusion term
    
    Methods
    ===========
    value: float return present value of call option
    vega: float return vega of call option
    imp_vol: float return implied volatility given option quote
    '''
    
    def __init__(self, S0, K, t, M, r, sigma):
        self.S0 = float(S0)
        self.K = K
        self.t = t
        self.M = M
        self.r = r
        self.sigma = sigma

    def update_ttm(self):
        '''Updates time-to-maturity self.T.'''
        if self.t > self.M:
            raise ValueError("Pricing date later than maturity.")
        self.T = (self.M-self.t).days/365

#     def dl(self):
#         '''Helper function.'''
#         d1 = ((log(self.S0/self.K)) + (self.r+0.5*self.sigma **2)*self.T)/(self.sigma*np.sqrt(self.T))
#         return d1

    def value(self):
        '''Return option value.'''
        self.update_ttm()
        d1 = ((np.log(self.S0/self.K)) + (self.r+0.5*self.sigma **2)*self.T)/(self.sigma*np.sqrt(self.T))
        d2 = ((np.log(self.S0/self.K)+(self.r-0.5*self.sigma**2)*self.T)/(self.sigma*np.sqrt(self.T)))
        value = (self.S0*stats.norm.cdf(d1, 0.0, 1.0)-self.K*np.exp(-self.r*self.T)*stats.norm.cdf(d2, 0.0, 1.0))
        return value

    def vega(self):
        '''Return of option.'''
        self.update_ttm()
        d1 = self.d1
        vega = self.S0*stats.norm.pdf(d1, 0.0, 1.0)*np.sqrt(self.T)
        return vega

    def imp_vol(self, C0, sigma_est=0.2):
        '''Return implied volatility given option price.'''
        option = call_option(self.S0, self.K, self.t, self.M, self.r, sigma_est)
        option.update_ttm()
        def difference(sigma):
            option.sigma = sigma
            return option.value() - C0
        iv = fsolve(difference, sigma_est)[0]
        return iv