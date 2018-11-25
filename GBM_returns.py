import numpy as np
import pandas as pd
import scipy.stats as scs
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.family'] = 'serif'

#
# Simulate a Number of Years of Daily Stock Quotes
#

def simulate_gbm():
    
    # simulation parameters
    np.random.seed(250000)
    gbm_dates = pd.DatetimeIndex(start = '01-01-2009', end = '30-09-2014', freq = 'B')
    M = len(gbm_dates)  # time steps
    I = 1               # index level paths
    dt = 1/252.         # fixed for simplicity
    df = np.exp(-r*dt)  # discount factor

    # stock price paths
    rand = np.random.standard_normal((M, I)) # random numbers
    S = np.zeros_like(rand)
    S[0] = S0           # initial values
    for t in range(1, M):
        S[t] = S[t -1]*np.exp((r-vol**2/2)*dt+vol*rand[t]*np.sqrt(dt))

    gbm = pd.DataFrame(S[:, 0], index=gbm_dates, columns=['index'])
    gbm['returns'] = np.log(gbm['index']/gbm['index'].shift(1))

    # Realized Volatility (eg. as defined for variance swaps
    gbm['rea_var'] = 252 *np.cumsum(gbm['returns']**2)/np.arange(len(gbm))
    gbm['rea_vol'] = np.sqrt(gbm['rea_var'])
    gbm = gbm.dropna()
    return gbm

# Return Sample Statistics and Normality Test

def quotes_statistics(data):
    print "RETURN SAMPLE STATISTICS"
    print "---------------------------------"
    print "Mean of Daily Log Returns %9.6f" % np.mean(data['returns'])
    print "Std of Daily Log Returns %9.6f" % np.std(data['returns'])
    print "Mean of Annua. Log Returns %9.6f" % (np.mean(data['returns'])*252)
    print "Std of Annua. Log Returns %9.6f" % (np.std(data['returns'])*np.sqrt(252))
    print "---------------------------------"
    print "Skew of Sample Log Returns %9.6f" % scs.skew(data['returns'])
    print "Skew of Sample Log Returns %9.6f" % scs.skewtest(data['returns'])[1]
    print "---------------------------------"
    print "Kurt of Sample Log Return %9.6f" % scs.kurtosis(data['returns'])
    print "Kurt Normal Test p-value %9.6f" % scs.kurtosistest(data['returns'])[1]
    print "---------------------------------"
    print "Normal Test p-value %9.6f" % scs.normaltest(data['returns'])[1]
    print "---------------------------------"
    print "Realized Volatility %9.6f" % data['rea_vol'][-1]
    print "Realized Variance %9.6f" % data['rea_var'][-1]

#
# Grapraphical Output
#


# daily quotes and log returns
def quotes_returns(data):
    '''Plots quotes and returns.'''
    plt.figure(figsize=(9, 6))
    plt.subplot(211)
    data['index'].plot()
    plt.ylabel('daily quotes')
    plt.grid(True)
    plt.axis('tight')

    plt.subplot(212)
    data['returns'].plot()
    plt.ylabel('daily log returns')
    plt.grid(True)
    plt.axis('tight')
        
    
# histogram of annualized daily log returns
def return_histogram(data):
    '''Plots a histogram of the returns'''
    plt.figure(figsize=(9, 5))
    x = np.linspace(np.min(data['returns']), np.max(data['returns']), 100 )
    plt.hist(np.array(data['returns']), bins=50, normed=True)
    y = norm.pdf(x, np.mean(data['returns']), np.std(data['returns']))
    plt.plot(x, y, linewidth=2)
    plt. xlabel('log return')
    plt.ylabel('frequency/ probability')
    plt.grid(True)

#Q-Q plot of annualized daily log returns
def return_qqplot(data):
    '''Generates a Q-Qplot of the returns'''
    plt.figure(figsize=(9, 5))
    sm.qqplot(data['returns'], line='s')
    plt.grid(True)
    plt.xlabel('theoretical quantile')
    plt.ylabel('sample quantiles')


# realized volatility
def realized_volatility(data):
    '''Plots the realized volatility.'''
    plt.figure(figsize=(9, 5))
    data['rea_vol'].plot()
    plt.ylabel('realized volatility')
    plt.grid(True)
    
    
# mean return, volatility and correlation (252 days moving = 1 year)
def rolling_statistics(data):
    '''Calculates and plots rollings statistics (mean, std, correlation).'''
    plt.figure(figsize=(11, 8))

    plt.subplot(311)
    mr = pd.rolling_mean(data['returns'], 252)*252
    mr.plot()
    plt.grid(True)
    plt.ylabel('returns(252d)')
    plt.axhline(mr.mean(), color='r', ls= 'dashed', lw=1.5)

    plt.subplot(312)
    vo = pd.rolling_std(data['returns'], 252)*np.sqrt(252)
    vo.plot()
    plt.grid(True)
    plt.ylabel('volatility (252d)')
    plt.axhline(vo.mean(), color='r', ls='dashed', lw=1.5)
    vx=plt.axis()

    plt.subplot(313)
    corr=pd.rolling_corr(mr, vo, 252)
    corr.plot()
    plt.grid(True)
    cx=plt.axis()
    plt.axis([vx[0], vx[1], cx[2], cx[3]])
    plt.axhline(corr.mean(), color='r', ls='dashed', lw=1.5)