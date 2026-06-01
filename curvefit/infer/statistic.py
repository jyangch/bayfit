from decimal import Decimal
from math import factorial
from types import MappingProxyType

import numpy as np


class Statistic:
    @staticmethod
    def chi_square(mo_func, params, x, y, x_err, y_err, w, up):

        x, y = np.array(x), np.array(y)
        x_err, y_err = np.array(x_err), np.array(y_err)
        my = mo_func(x, params)

        yerr = (y < my).astype(int) * y_err[1] + (y >= my).astype(int) * y_err[0]
        S = w * (y - my) ** 2 / yerr**2
        S[(y >= my) & up] = 0
        S[(y < my) & up] = np.inf
        return -0.5 * np.sum(S)

    @staticmethod
    def chi_square_full(mo_func, params, x, y, x_err, y_err, w, up):

        x, y = np.array(x), np.array(y)
        x_err, y_err = np.array(x_err), np.array(y_err)
        my = mo_func(x, params[:-1])
        logv = params[-1]

        yerr = (y < my).astype(int) * y_err[1] + (y >= my).astype(int) * y_err[0]
        sigma2 = yerr**2 + 10 ** (2 * logv)
        S = w * ((y - my) ** 2 / sigma2 + np.log(2 * np.pi * sigma2))
        S[(y >= my) & up] = 0
        S[(y < my) & up] = np.inf
        return -0.5 * np.sum(S)

    @staticmethod
    def log_chi_square(mo_func, params, x, y, x_err, y_err, w, up):

        x, y = np.array(x), np.array(y)
        x_err, y_err = np.array(x_err), np.array(y_err)
        logy = np.log10(y)
        logy_err = [logy - np.log10(y - y_err[0]), np.log10(y + y_err[1]) - logy]
        my = mo_func(x, params)
        logmy = np.log10(my)

        logyerr = (
            np.array(logy < logmy).astype(int) * logy_err[1]
            + np.array(logy >= logmy).astype(int) * logy_err[0]
        )
        S = w * (logy - logmy) ** 2 / logyerr**2
        S[(logy >= logmy) & up] = 0
        S[(logy < logmy) & up] = np.inf
        return -0.5 * np.sum(S)

    @staticmethod
    def vdr(mo_func, params, x, y, x_err, y_err, w, up):

        x, y = np.array(x), np.array(y)
        x_err, y_err = np.array(x_err), np.array(y_err)
        my = mo_func(x, params)

        k, log_v = params[0], params[2]
        if k > 0:
            xerr = (y < my).astype(int) * x_err[0] + (y >= my).astype(int) * x_err[1]
        else:
            xerr = (y < my).astype(int) * x_err[1] + (y >= my).astype(int) * x_err[0]
        yerr = (y < my).astype(int) * y_err[1] + (y >= my).astype(int) * y_err[0]
        sigma2 = np.exp(2 * log_v) + k**2 * xerr**2 + yerr**2
        S = w * ((y - my) ** 2 / sigma2 + np.log(2 * np.pi * sigma2))
        S[(y >= my) & up] = 0
        S[(y < my) & up] = np.inf
        return -0.5 * np.sum(S)

    @staticmethod
    def odr(mo_func, params, x, y, x_err, y_err, w, up):

        x, y = np.array(x), np.array(y)
        x_err, y_err = np.array(x_err), np.array(y_err)
        my = mo_func(x, params)

        k, log_v = params[0], params[2]
        if k > 0:
            xerr = (y < my).astype(int) * x_err[0] + (y >= my).astype(int) * x_err[1]
        else:
            xerr = (y < my).astype(int) * x_err[1] + (y >= my).astype(int) * x_err[0]
        yerr = (y < my).astype(int) * y_err[1] + (y >= my).astype(int) * y_err[0]

        delta2 = (y - my) ** 2 / (k**2 + 1)
        sigma2 = (k**2 * xerr**2 + yerr**2) / (k**2 + 1) + np.exp(2 * log_v)
        S = w * (delta2 / sigma2 + np.log(2 * np.pi * sigma2))
        S[(y >= my) & up] = 0
        S[(y < my) & up] = np.inf
        return -0.5 * np.sum(S)

    @staticmethod
    def groth(mo_func, params, x, y, x_err, y_err, w, up):

        x, y = np.array(x), np.array(y)
        x_err, y_err = np.array(x_err), np.array(y_err)
        my = mo_func(x, params)

        lnL = 0
        for i in range(len(y)):
            yi = Decimal(float(y[i]))
            myi = Decimal(float(my[i]))

            m = 0
            dft = 0
            while True:
                dfti = (yi**m) * (myi**m) / (factorial(m)) ** 2
                if dfti > 1e-20:
                    dft += dfti
                    m += 1
                else:
                    break
            Li = np.exp(-(float(yi) + float(myi))) * float(dft)
            lnL += np.log(Li)
        return lnL

    _allowed_stats = MappingProxyType(
        {
            'chi^2': chi_square.__func__,
            'chi^2f': chi_square_full.__func__,
            'logchi^2': log_chi_square.__func__,
            'vdr': vdr.__func__,
            'odr': odr.__func__,
            'groth': groth.__func__,
        }
    )
