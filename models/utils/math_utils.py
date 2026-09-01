import numpy as np


def almon_weights(theta1, theta2, max_lag=4):
    lags = np.arange(0, max_lag)
    exponents = theta1 * lags + theta2 * (lags**2)
    exponents_shifted = exponents - np.max(exponents)
    unnormalized_weights = np.exp(exponents_shifted)
    return unnormalized_weights / np.sum(unnormalized_weights)


def midas_loss_univariate(params, X, y):
    alpha, beta, t1, t2 = params[0], params[1], params[2], params[3]
    w = almon_weights(t1, t2, max_lag=X.shape[1])
    y_pred = alpha + beta * np.dot(X, w)
    return np.sum((y - y_pred) ** 2)


def midas_loss_bivariate(params, X1, X2, y):
    alpha = params[0]
    beta1 = params[1]
    beta2 = params[2]

    theta1_x1, theta2_x1 = params[3], params[4]
    theta1_x2, theta2_x2 = params[5], params[6]

    w1 = almon_weights(theta1_x1, theta2_x1, max_lag=X1.shape[1])
    w2 = almon_weights(theta1_x2, theta2_x2, max_lag=X2.shape[1])

    X1_agg = np.dot(X1, w1)
    X2_agg = np.dot(X2, w2)

    y_pred = alpha + (beta1 * X1_agg) + (beta2 * X2_agg)
    return np.sum((y - y_pred) ** 2)


def midas_loss_trivariate(params, X1, X2, X3, y):
    alpha = params[0]
    beta1, beta2, beta3 = params[1], params[2], params[3]

    theta1_g, theta2_g = params[4], params[5]
    theta1_b, theta2_b = params[6], params[7]
    theta1_y, theta2_y = params[8], params[9]

    w_gas = almon_weights(theta1_g, theta2_g, max_lag=X1.shape[1])
    w_bcpi = almon_weights(theta1_b, theta2_b, max_lag=X2.shape[1])
    w_yield = almon_weights(theta1_y, theta2_y, max_lag=X3.shape[1])

    X_gas_agg = np.dot(X1, w_gas)
    X_bcpi_agg = np.dot(X2, w_bcpi)
    X_yield_agg = np.dot(X3, w_yield)

    y_pred = alpha + (beta1 * X_gas_agg) + (beta2 * X_bcpi_agg) + (beta3 * X_yield_agg)
    return np.sum((y - y_pred) ** 2)


def ar_midas_loss_univariate(params, y_lag, X, y):
    alpha, rho, beta, t1, t2 = params[0], params[1], params[2], params[3], params[4]
    w = almon_weights(t1, t2, max_lag=X.shape[1])
    y_pred = alpha + (rho * y_lag) + (beta * np.dot(X, w))
    return np.sum((y - y_pred) ** 2)


def ar_midas_loss_bivariate(params, y_lag, X1, X2, y):
    alpha, rho, beta1, beta2 = params[0], params[1], params[2], params[3]
    t1_1, t2_1, t1_2, t2_2 = params[4], params[5], params[6], params[7]

    w1 = almon_weights(t1_1, t2_1, max_lag=X1.shape[1])
    w2 = almon_weights(t1_2, t2_2, max_lag=X2.shape[1])

    y_pred = alpha + (rho * y_lag) + (beta1 * np.dot(X1, w1)) + (beta2 * np.dot(X2, w2))
    return np.sum((y - y_pred) ** 2)
