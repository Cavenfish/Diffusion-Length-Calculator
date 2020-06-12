import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def EBICurrent(d, A, L):
    return A*d**(-1/2)*np.exp(-d/L)

def linear(x, m, b):
    return m*x + b

def actE(c, E, A):
    return A* np.exp(-E/c)

def get_charge_density(I, A, times):
    q = []
    for t in times:
        q.append((I*t)/A)
    return q

def fit_data(file, op, sp, fp, tp, save_name=None, neg=1):
    x  = 'x(um)'
    y  = 'EBIC(V)'
    #Read in data
    df     = pd.read_csv(file, sep='\t', header=12)
    indexn = df[df[x] > 10].index
    df.drop(indexn, inplace=True)

    #Get X and Y values from data
    X      = df[x].values
    Y      = df[y].values * neg * 20e-9
    X2     = X - op
    Y2     = Y - np.mean(Y[ (X > tp) & (X < tp + 1) ])
    yvals  = Y - np.mean(Y[ (X > tp) & (X < tp + 1) ])
    yvals  = yvals[(X > op) & (X < fp)]
    xvals  = X[(X > op) & (X < fp)]
    xvals  = xvals - op
    logI   = np.log(yvals * np.sqrt(xvals))

    X2fit      = xvals[xvals > sp]
    Y2fit      = logI[xvals > sp]
    valid      = ~( np.isnan(Y2fit) | np.isinf(Y2fit) )
    popt, pcov = curve_fit(linear, X2fit[valid], Y2fit[valid])
    yfit       = linear(xvals, *popt)

    dl = round(-1 * popt[0]**-1, 2)

    if save_name is False:
        return dl

    s  = 'Diffusion Length: ' + str(dl) + 'μm'

    figs, axs = plt.subplots(3,1)
    df.plot(x,y, ax=axs[0], legend=False)
    axs[0].plot(X[ (X > tp) & (X < tp + 1) ],
                neg* Y[ (X > tp) & (X < tp + 1) ]/20e-9,
                 lw=5, alpha=80, color='orange')
    axs[1].plot(X2,Y2)
    figs.suptitle('Plot for: ' +file.split('/')[2].strip('.txt'))
    axs[1].plot(X2[(X2 > sp) & (X2 < fp-op)],
                Y2[(X2 > sp) & (X2 < fp-op)], lw=5, alpha=80, color='yellow')
    axs[2].plot(xvals[1:], logI[1:], label='Data')
    axs[2].plot(xvals, yfit, ls='--', color='red', label='Fitting')
    axs[2].plot(xvals[xvals > sp], logI[xvals > sp], color='yellow',
                alpha=90, lw=5)
    plt.text(0, min(yfit), s)
    #plt.legend()
    plt.xlabel('Distance (μm)')
    plt.ylabel(r'ln(I$x^{-α}$)')
    plt.tight_layout()
    figs.subplots_adjust(top=0.88)

    if save_name is None:
        plt.show()
        return
    else:
        plt.savefig(save_name)
        plt.close()
        plt.plot(xvals[xvals > sp], logI[xvals > sp], label='Data')
        plt.plot(xvals, yfit, ls='--', color='red', label='Fitting')
        plt.savefig(save_name.replace('.png', '--2.png'))
        plt.close()

    return dl


def fit_LvT(L, T, sig, save_name=None):
    k          = 1.38064852e-23
    c          = 2*k* np.array(T)
    popt, pcov = curve_fit(actE, c, L, p0=(13e-26, 1))
    xfit       = np.arange(50, 325, 2.75)
    cfit       = 2*k* np.array(xfit)
    yfit       = actE(cfit, *popt)

    logL       = np.log(L)
    popt, pcov = curve_fit(linear, c**-1, logL)
    yfit2      = linear(c**-1, *popt)

    aE         = -popt[0] * 6.241509e18 * 1e3
    print(aE)

    plt.errorbar(T, L, sig, label='Data', fmt='o')
    plt.plot(xfit, yfit, label='Fitting', color='red', ls='--')
    plt.title('Diffusion Length vs Temperature')
    plt.ylabel(r'Diffusion Length ($μ$m)')
    plt.xlabel('Temperatrue (K)')

    if save_name is None:
        plt.show()
        return
    else:
        plt.savefig(save_name)
        plt.close()
        plt.scatter(c**-1, logL, label='Data')
        plt.plot(c**-1, yfit2, ls='--', color='red', label='Fitting')
        plt.xlabel(r'$\frac{1}{k_b T}$', fontsize=15)
        plt.ylabel(r'$\ln{L}$', fontsize=15)
        plt.savefig(save_name.replace('.png', '--2.png'))
        plt.close()

    return
