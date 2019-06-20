
import numpy as np
from scipy.optimize import minimize
import scipy.stats as sps
from matplotlib import pyplot as plt

def cut01(ans):
    if np.isscalar(ans):
        return np.clip(ans, 1e-4, 1 - 1e-4)
    ans[ans <= 1e-4] = 1e-4
    ans[ans >= 1 - 1e-4] = 1 - 1e-4
    return ans

def bayesian_solve(alpha0, beta0, train, L):
    ans = (train + alpha0 - 1) / (L + alpha0 + beta0 - 2)
    return cut01(ans)

def calc_loglike(p_pred, val, L):
    if (p_pred >= 1).sum() + (p_pred <= 0).sum() > 0:
        print (p_pred)
        print ('aaaaaa!!!!1111')
    return (val * np.log(p_pred)).sum() + ((L - val) * np.log(1 - p_pred)).sum()

def calc_bayesian_solve_loglike(alpha0, beta0, train, val, L, use_prior=False):
    p_pred = bayesian_solve(alpha0, beta0, train, L)
    ans = calc_loglike(p_pred, val, L)
    if use_prior:
        m = alpha0 / (alpha0 + beta0)
        s = alpha0 + beta0
        ans += sps.beta.logpdf(m, 1, 10)
        ans += sps.norm(4, 1).logpdf(np.log(s))
    return ans

def stupid_solution(train, val, L):
    return cut01((train + val) / 2 / L)

def calc_llp(p_pred, train, test, L):
    p_pred = cut01(p_pred)
    p_const = cut01(train.sum() / L / train.shape[0])
    return (calc_loglike(p_pred, test, 1e6) - calc_loglike(p_const, test, 1e6)) / 1e6 / test.shape[0]

def evaluate(solution_fun, train, val, test, L):
    p_pred = solution_fun(train, val, L)
    if np.any(p_pred == 0):
        raise Exception("Zero predicted probability")
        

    #print ("p_pred", p_pred[:5])
    #print (test[:5] / 1e6)
    #plt.hist(test / 1e6, bins=20, density=True)
    #plt.show()
    return calc_llp(p_pred, train + val, test, 2 * L)

def max_loglike_solution(train, val, L):
    loglike = lambda x: calc_bayesian_solve_loglike(x[0], x[1], train, val, L)
    try:
        res = minimize(fun=lambda x: -loglike(1 + np.exp(x)), x0=(1, 1))
    except FloatingPointError as e:
        return mean_value_solution(train, val, L)
    if not res.success:
        return mean_value_solution(train, val, L)
    else:
        return bayesian_solve(1 + np.exp(res.x[0]), 1 + np.exp(res.x[1]), train + val, L * 2)
    
def mean_value_solution(train, val, L):
    m = ((train + val) / 2. / L).mean()
    m = cut01(m)
    sum_ab = 1 / m * 20
    return bayesian_solve(m * sum_ab, (1 - m) * sum_ab, train + val, 2 * L)

def calibration_curve_solution(train, val, L):
    try:
        line = sps.linregress((val) / L, (train) / L, )
    except FloatingPointError as e:
        return max_loglike_solution(train, val, L)
    if line.slope == 0:
        return max_loglike_solution(train, val, L)
    tg = 1 / line.slope
    sum_ab = tg * L - L + 2
    if sum_ab == 0:
        return max_loglike_solution(train, val, L)
    m = (L * line.intercept / line.slope + 1) / sum_ab
    if tg != tg or m * sum_ab < 1 or sum_ab > 1e5 or m > 1:
        return max_loglike_solution(train, val, L)
    
    space = np.linspace(0, .2, 100)
    return bayesian_solve(m * sum_ab, sum_ab * (1 - m), train + val, 2 * L)

def mean_std_prior(p):
    m = p.mean()
    sq_std = p.std() ** 2
    if sq_std == 0:
        return None, None
    sum_ab = m * (1 - m) / sq_std
    if m * sum_ab < 1 or sum_ab != sum_ab:
        return None, None
    return m * sum_ab, (1 - m) * sum_ab

def mean_std_value_solution(train, val, L, iters=2):
    alpha, beta = mean_std_prior((train + val) / 2 / L)
    
    for i in range(iters):
        if alpha is None:
            return mean_value_solution(train, val, L)
        p = bayesian_solve(alpha, beta, train + val, 2 * L)
        alpha, beta = mean_std_prior(p)

    if alpha is None:
        return mean_value_solution(train, val, L)
    
    return bayesian_solve(alpha, beta, train + val, 2 * L)
