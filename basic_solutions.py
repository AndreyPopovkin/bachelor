
import numpy as np
from scipy.optimize import minimize
import scipy.stats as sps

def cut01(ans):
    ans[ans <= 1e-4] = 1e-4
    ans[ans >= 1 - 1e-4] = 1 - 1e-4
    return ans

def bayesian_solve(alpha0, beta0, train, L):
    ans = (train + alpha0 - 1) / (L + alpha0 + beta0 - 2)
    return cut01(ans)

def calc_loglike(p_pred, val, L):
    return (val * np.log(p_pred)).sum() + ((L - val) * np.log(1 - p_pred)).sum()

def calc_bayesian_solve_loglike(alpha0, beta0, train, val, L):
    p_pred = bayesian_solve(alpha0, beta0, train, L)
    return calc_loglike(p_pred, val, L)

def stupid_solution(train, val, L):
    return (train + val + 1e-4) / 2 / (L + 2e-4)

def calc_llp(p_pred, train, test, L):
    p_const = train.sum() / L / train.shape[0]
    if p_const < 1e-4:
        p_const = 1e-4
    return (calc_loglike(p_pred, test, 1e6) - calc_loglike(p_const, test, 1e6)) / 1e6 / test.shape[0]

def evaluate(solution_fun, train, val, test, L):
    p_pred = solution_fun(train, val, L)
    if np.any(p_pred == 0):
        raise Exception("Zero predicted probability")
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
    if m < 1e-4:
        m = 1e-4
    sum_ab = 1 / m * 20
    return bayesian_solve(m * sum_ab, (1 - m) * sum_ab, train + val, 2 * L)

def calibration_curve_solution(train, val, L):
    try:
        line = sps.linregress((val) / L, (train) / L, )
    except FloatingPointError as e:
        return mean_value_solution(train, val, L)
    if line.slope == 0:
        return mean_value_solution(train, val, L)
    tg = 1 / line.slope
    sum_ab = tg * L - L + 2
    m = ((train + val) / 2 / L).mean()
    if tg != tg or m * sum_ab < 1 or sum_ab > 1e5:
        return mean_value_solution(train, val, L)
    
    return bayesian_solve(m * sum_ab, sum_ab * (1 - m), train + val, 2 * L)
    
def mean_std_value_solution(train, val, L):
    m = ((train + val) / 2. / L).mean()
    sq_std = ((train + val) / 2. / L).std() ** 2
    if sq_std == 0:
        return mean_value_solution(train, val, L)
    sum_ab = m * (1 - m) / sq_std
    if m * sum_ab < 1 or sum_ab != sum_ab:
        return mean_value_solution(train, val, L)
    return bayesian_solve(m * sum_ab, sum_ab * (1 - m), train + val, 2 * L)
    
def max_loglike_solution2(train, val, L):
    loglike = lambda x: calc_bayesian_solve_loglike(x[0], x[1], train, val, L)
    try:
        res = minimize(fun=lambda x: -loglike(1 + np.exp(x)), x0=(1, 1))
    except FloatingPointError as e:
        return mean_value_solution(train, val, L)
    if not res.success:
        return mean_value_solution(train, val, L)
    else:
        return bayesian_solve(1 + np.exp(res.x[0]), 1 + np.exp(res.x[1]), train, L)
