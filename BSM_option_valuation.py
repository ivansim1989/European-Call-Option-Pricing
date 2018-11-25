import math
import numpy as np
from scipy.stats import norm
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.family'] = 'serif'

#
# Valuation Functions
#


def BSM_call_value(St, K, T, r, sigma):
    ''' Calculates Black-Scholes-Merton European call option value.
    Parameters
    ==========
    St : float
        stock/index level at time t
    K : float
        strike price
    T : float
        date of maturity/time-to-maturity
    r : float
        constant, risk-less short rate
    sigma : float
        volatility
    Returns
    =======
    call_value : float
        European call present value at t
    '''
    N = norm.cdf                    # Cumulative density function of standard normal random variable x.
    disc = np.exp(-r*T)             # Discounting
    vol = sigma*np.sqrt(T)          # Time-scaled volatility

    # calculate price analytically using Black-Scholes
    d1 = ((np.log(St/K)+(r+0.5*sigma**2)*T))/vol
    d2 = d1-vol
    call_value = St*N(d1)-disc*K*N(d2)
    
    return call_value
    # N(d1) is the present value of receiving the stock if and only if the option finishes in the money, and the discounted exercise payment times.
    # N(d2) is the present value of paying the exercise price in that event.
    


def BSM_put_value(St, K, T, r, sigma):
    ''' Calculates Black-Scholes-Merton European put option value.
    Parameters
    ==========
    St : float
        stock/index level at time t
    K : float
        strike price
    T : float
        date of maturity/time-to-maturity
    r : float
        constant, risk-less short rate
    sigma : float
        volatility
    Returns
    =======
    put_value : float
        European put present value at t
    '''
    put_value = disc*K*N(-d2)-St*N(-d1)
    return put_value


#
# Plotting European Option Values
#


def plot_values(function):
    ''' Plots European option values for different parameters c.p. '''
    plt.figure(figsize=(10, 8.3))
    points = 100
    #
    # Model Parameters
    #
    St = 100.0  # index level
    K = 100.0  # option strike
    t = 0.0  # valuation date
    T = 1.0  # maturity date
    r = 0.05  # risk-less short rate
    sigma = 0.2  # volatility

    # C(K) plot
    plt.subplot(221)
    klist = np.linspace(80, 120, points)
    vlist = [function(St, K, T, r, sigma) for K in klist]
    plt.plot(klist, vlist)
    plt.grid()
    plt.xlabel('strike $K$')
    plt.ylabel('present value')

    # C(T) plot
    plt.subplot(222)
    tlist = np.linspace(0.0001, 1, points)
    vlist = [function(St, K, T, r, sigma) for T in tlist]
    plt.plot(tlist, vlist)
    plt.grid(True)
    plt.xlabel('maturity $T$')

    # C(r) plot
    plt.subplot(223)
    rlist = np.linspace(0, 0.1, points)
    vlist = [function(St, K, T, r, sigma) for r in rlist]
    plt.plot(tlist, vlist)
    plt.grid(True)
    plt.xlabel('short rate $r$')
    plt.ylabel('present value')
    plt.axis('tight')

    # C(sigma) plot
    plt.subplot(224)
    slist = np.linspace(0.01, 0.5, points)
    vlist = [function(St, K, T, r, sigma) for sigma in slist]
    plt.plot(slist, vlist)
    plt.grid(True)
    plt.xlabel('volatility $\sigma$')
    plt.tight_layout()