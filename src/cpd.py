"""Changepoint detection via Gaussian Processes.

Implements the Matern 3/2 kernel fit and the changepoint kernel from
Wood, Roberts & Zohren (2022), "Slow Momentum with Fast Reversion".

All GP computations use scipy and numpy only -- no GPflow / TensorFlow.

Reference equations from the paper:
    Eq. 4  -- Matern 3/2 kernel
    Eq. 7  -- Negative log marginal likelihood (NLML)
    Eq. 9  -- Sigmoid-blended changepoint kernel
    Eq. 10 -- Severity (nu) and location (gamma)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
import ruptures as rpt

# ---------------------------------------------------------------------------
# Kernel functions
# ---------------------------------------------------------------------------


def _matern32_kernel(X: np.ndarray, sigma_f: float, lengthscale: float) -> np.ndarray:
    """Matern 3/2 covariance matrix.  (Paper Eq. 4)

    k(x, x') = sigma_f^2 * (1 + sqrt(3)*|x - x'| / l)
                          * exp(-sqrt(3)*|x - x'| / l)

    Parameters
    ----------
    X : (n,) array of input locations (time indices).
    sigma_f : output-scale standard deviation (sigma_n in paper Eq. 4,
              but we call it sigma_f to avoid confusion with noise).
    lengthscale : length-scale lambda.

    Returns
    -------
    K : (n, n) covariance matrix.
    """
    dist = np.abs(X[:, None] - X[None, :])          # |x - x'|
    lengthscale = max(lengthscale, 1e-10)            # guard against zero
    r = np.sqrt(3.0) * dist / lengthscale
    K = sigma_f ** 2 * (1.0 + r) * np.exp(-r)
    np.nan_to_num(K, copy=False, nan=0.0, posinf=1e10, neginf=0.0)
    return K


def _sigmoid(x: np.ndarray, c: float, s: float) -> np.ndarray:
    """Logistic sigmoid used for the changepoint blend.

    sigma(x) = 1 / (1 + exp(-s * (x - c)))

    where c is the changepoint location and s > 0 is the steepness.
    (Paper: sigma(x) = 1/(1 + e^{-s(x-c)}), see text below Eq. 8.)

    Returns values in (0, 1).
    """
    z = np.asarray(s * (x - c), dtype=np.float64)
    # Numerically stable sigmoid: clamp the argument to avoid overflow
    z_clamped = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z_clamped))


def _changepoint_kernel(
    X: np.ndarray,
    sigma_f1: float, l1: float,
    sigma_f2: float, l2: float,
    c: float, s: float,
) -> np.ndarray:
    """Changepoint kernel.  (Paper Eq. 9)

    k_cp(x, x') = k_{s1}(x, x') * sig(x) * sig(x')
                 + k_{s2}(x, x') * sig_bar(x) * sig_bar(x')

    where sig_bar(x) = 1 - sig(x).

    Each of k_{s1}, k_{s2} is a Matern 3/2 kernel with its own
    (sigma_f, lengthscale) pair.
    """
    K1 = _matern32_kernel(X, sigma_f1, l1)
    K2 = _matern32_kernel(X, sigma_f2, l2)

    sig = _sigmoid(X, c, s)                      # (n,)
    sig_bar = 1.0 - sig

    # Outer products for the blending weights
    S = sig[:, None] * sig[None, :]               # sig(x) * sig(x')
    S_bar = sig_bar[:, None] * sig_bar[None, :]   # sig_bar(x) * sig_bar(x')

    return K1 * S + K2 * S_bar


# ---------------------------------------------------------------------------
# Negative log marginal likelihood
# ---------------------------------------------------------------------------


def _nlml(K: np.ndarray, y: np.ndarray, sigma_n: float) -> float:
    """Negative log marginal likelihood of a GP.  (Paper Eq. 7)

    nlml = 0.5 * y^T V^{-1} y  +  0.5 * log|V|  +  n/2 * log(2*pi)

    where  V = K + sigma_n^2 * I.

    Uses Cholesky decomposition for numerical stability.

    Parameters
    ----------
    K : (n, n) covariance matrix from the kernel.
    y : (n,) observation vector (standardized returns).
    sigma_n : observation-noise standard deviation.

    Returns
    -------
    nlml : scalar, the negative log marginal likelihood.
    """
    n = len(y)
    V = K + sigma_n ** 2 * np.eye(n)

    # Add a small jitter for numerical stability of Cholesky
    jitter = 1e-6
    V += jitter * np.eye(n)

    try:
        L = np.linalg.cholesky(V)
    except np.linalg.LinAlgError:
        # If Cholesky fails, return a very large penalty
        return 1e10

    # Solve L alpha_tmp = y, then L^T alpha = alpha_tmp  =>  alpha = V^{-1} y
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))

    # 0.5 * y^T V^{-1} y
    data_fit = 0.5 * y @ alpha

    # 0.5 * log|V| = sum(log(diag(L)))  (since |V| = |L|^2)
    complexity = np.sum(np.log(np.diag(L)))

    # n/2 * log(2*pi)
    constant = 0.5 * n * np.log(2.0 * np.pi)

    return data_fit + complexity + constant


# ---------------------------------------------------------------------------
# Fitting routines
# ---------------------------------------------------------------------------


def _fit_base_matern(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Fit a GP with a single Matern 3/2 kernel by minimizing NLML.

    Hyperparameters: theta = [log(sigma_f), log(lengthscale), log(sigma_n)]
    We optimize in log-space so that all parameters remain positive.

    Initialization (from paper, page 9):
        All Matern 3/2 kernel hyperparameters are initialized to 1.

    Returns
    -------
    best_nlml : the minimized negative log marginal likelihood.
    best_params : array [sigma_f, lengthscale, sigma_n] at the optimum.
    """
    n = len(y)

    def objective(log_theta):
        sigma_f, lengthscale, sigma_n = np.exp(log_theta)
        K = _matern32_kernel(X, sigma_f, lengthscale)
        return _nlml(K, y, sigma_n)

    # Initialize all kernel hyperparameters to 1  (paper p.9)
    log_theta0 = np.array([0.0, 0.0, 0.0])  # log(1) = 0
    # Bound log-params to avoid overflow in exp()
    log_bounds = [(-10.0, 10.0)] * 3

    result = minimize(
        objective,
        log_theta0,
        method="L-BFGS-B",
        bounds=log_bounds,
        options={"maxiter": 200, "ftol": 1e-8},
    )

    best_params = np.exp(result.x)
    best_nlml = result.fun

    return best_nlml, best_params


def _fit_changepoint(
    X: np.ndarray,
    y: np.ndarray,
    base_params: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Fit a GP with the changepoint kernel by minimizing NLML.

    Hyperparameters (7 total):
        theta = [log(sigma_f1), log(l1),    -- kernel before changepoint
                 log(sigma_f2), log(l2),    -- kernel after changepoint
                 c,                         -- changepoint location (NOT in log-space)
                 log(s),                    -- steepness (log-space, s > 0)
                 log(sigma_n)]              -- noise std

    Initialization (from paper, page 9):
        - c = t - l/2  (middle of the window).  Since we use X = [0, ..., n-1],
          this is (n-1)/2.
        - s = 1.
        - k_{s1} and k_{s2} are initialized with the same values as the
          base Matern fit.
        - sigma_n is initialized from the base Matern fit.

    Constraint: c must lie within the window, i.e. c in (X[0], X[-1]).

    Returns
    -------
    best_nlml : the minimized negative log marginal likelihood.
    best_params : array [sigma_f1, l1, sigma_f2, l2, c, s, sigma_n]
                  at the optimum.
    """
    n = len(y)
    sigma_f_base, l_base, sigma_n_base = base_params

    # Initial changepoint location: middle of the window
    c0 = (n - 1) / 2.0
    s0 = 1.0

    # theta = [log(sf1), log(l1), log(sf2), log(l2), c, log(s), log(sn)]
    theta0 = np.array([
        np.log(sigma_f_base),   # log(sigma_f1) from base fit
        np.log(l_base),         # log(l1) from base fit
        np.log(sigma_f_base),   # log(sigma_f2) = same init
        np.log(l_base),         # log(l2) = same init
        c0,                     # c (raw, not log)
        np.log(s0),             # log(s)
        np.log(sigma_n_base),   # log(sigma_n) from base fit
    ])

    # Bounds: c must be within [X[0] + 0.5, X[-1] - 0.5] to stay inside window
    # Log-params bounded to avoid overflow in exp()
    bounds = [
        (-10.0, 10.0),                            # log(sigma_f1)
        (-10.0, 10.0),                            # log(l1)
        (-10.0, 10.0),                            # log(sigma_f2)
        (-10.0, 10.0),                            # log(l2)
        (float(X[0] + 0.5), float(X[-1] - 0.5)),  # c constrained to window
        (-10.0, 10.0),                            # log(s)
        (-10.0, 10.0),                            # log(sigma_n)
    ]

    def objective(theta):
        sigma_f1 = np.exp(theta[0])
        l1       = np.exp(theta[1])
        sigma_f2 = np.exp(theta[2])
        l2       = np.exp(theta[3])
        c        = theta[4]              # c is NOT in log-space
        s        = np.exp(theta[5])
        sigma_n  = np.exp(theta[6])

        K = _changepoint_kernel(X, sigma_f1, l1, sigma_f2, l2, c, s)
        return _nlml(K, y, sigma_n)

    result = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 300, "ftol": 1e-8},
    )

    # Extract optimized parameters
    theta_opt = result.x
    best_params = np.array([
        np.exp(theta_opt[0]),   # sigma_f1
        np.exp(theta_opt[1]),   # l1
        np.exp(theta_opt[2]),   # sigma_f2
        np.exp(theta_opt[3]),   # l2
        theta_opt[4],           # c
        np.exp(theta_opt[5]),   # s
        np.exp(theta_opt[6]),   # sigma_n
    ])
    best_nlml = result.fun

    return best_nlml, best_params


def _fit_changepoint_with_retry(
    X: np.ndarray,
    y: np.ndarray,
    base_params: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Fit the changepoint kernel, retrying with reset params if k_s1 == k_s2.

    From the paper (page 9):
        'In the rare case this process fails, we try again by
         reinitializing all changepoint kernel parameters to 1,
         with the exception of setting c = t - l/2.'
    """
    nlml_cp, cp_params = _fit_changepoint(X, y, base_params)

    # Check if the two sub-kernels collapsed to the same values
    # (sigma_f1 ~= sigma_f2 and l1 ~= l2)
    sf1, l1, sf2, l2 = cp_params[0], cp_params[1], cp_params[2], cp_params[3]
    if np.isclose(sf1, sf2, rtol=1e-3) and np.isclose(l1, l2, rtol=1e-3):
        # Retry with all parameters initialized to 1
        retry_params = np.array([1.0, 1.0, 1.0])  # sigma_f, l, sigma_n
        nlml_cp, cp_params = _fit_changepoint(X, y, retry_params)

    return nlml_cp, cp_params


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cpd_scores(returns, lbw: int) -> tuple[float, float]:
    """Return the (severity, location) pair (nu, gamma) for a lookback window.

    This is the main entry point.  Given a window of *raw* returns of length
    ``lbw``, the function:

    1. Standardizes the returns to zero mean and unit variance over the window.
    2. Fits a base GP with a single Matern 3/2 kernel  -> nlml_M.
    3. Fits a GP with the changepoint kernel             -> nlml_cp.
    4. Computes severity and location per Eq. 10:

        nu    = 1 - 1 / (1 + exp(-(nlml_cp - nlml_M)))
              = sigmoid(nlml_M - nlml_cp)

        gamma = (c - (t - l)) / l

       Since our X indices run from 0 to l-1 (the window is already
       extracted), the location simplifies to:

        gamma = c / (l - 1)

       (c = 0 means the changepoint is at the start of the window,
        c = l-1 means it is at the end.)

    Parameters
    ----------
    returns : array-like, shape (lbw,)
        Raw (or pre-standardized) returns over the lookback window.
        If you pass raw returns, they will be standardized internally.
    lbw : int
        Lookback window size in days (must equal len(returns)).

    Returns
    -------
    nu : float
        Severity in (0, 1).  Close to 1 = strong changepoint.
    gamma : float
        Location in (0, 1).  Close to 1 = changepoint near the end
        (most recent).
    """
    y = np.asarray(returns, dtype=np.float64).ravel()
    assert len(y) == lbw, f"len(returns)={len(y)} != lbw={lbw}"

    # ------------------------------------------------------------------
    # Step 1: Standardize returns over the window  (Paper Eq. 2)
    # r_hat = (r - mean(r)) / std(r)
    # ------------------------------------------------------------------
    mu = np.mean(y)
    std = np.std(y, ddof=0)
    if std < 1e-12:
        # Constant series -- no changepoint possible
        return 0.0, 0.5
    y_std = (y - mu) / std

    # Time indices: X = [0, 1, ..., lbw - 1]
    X = np.arange(lbw, dtype=np.float64)

    # ------------------------------------------------------------------
    # Step 2: Fit the base Matern 3/2 GP
    # ------------------------------------------------------------------
    nlml_base, base_params = _fit_base_matern(X, y_std)

    # ------------------------------------------------------------------
    # Step 3: Fit the changepoint GP
    # ------------------------------------------------------------------
    nlml_cp, cp_params = _fit_changepoint_with_retry(X, y_std, base_params)

    # ------------------------------------------------------------------
    # Step 4: Compute severity (nu) and location (gamma)  (Paper Eq. 10)
    # ------------------------------------------------------------------
    # nu = 1 - 1 / (1 + exp(-(nlml_cp - nlml_M)))
    #    = sigmoid(nlml_M - nlml_cp)
    #
    # When nlml_cp < nlml_M (changepoint kernel fits better), the argument
    # is positive, so nu -> 1.
    # When nlml_cp >= nlml_M (no improvement), the argument is <= 0,
    # so nu -> 0.
    delta = nlml_base - nlml_cp
    nu = 1.0 - 1.0 / (1.0 + np.exp(-(-delta)))  # = sigmoid(delta)
    # Simplifies to: nu = 1 / (1 + exp(-delta))
    # but we keep the paper's form for clarity.

    # Location: gamma = c / (lbw - 1), since our window indices are 0..lbw-1.
    # This maps the changepoint position to (0, 1).
    c_opt = cp_params[4]
    gamma = c_opt / (lbw - 1) if lbw > 1 else 0.5

    # Clip to (0, 1) for safety (c is bounded but numerical issues can arise)
    nu = np.clip(nu, 0.0, 1.0)
    gamma = np.clip(gamma, 0.0, 1.0)

    return float(nu), float(gamma)


# ---------------------------------------------------------------------------
# Convenience wrappers (kept for backward compatibility with project layout)
# ---------------------------------------------------------------------------


def fit_matern(returns):
    """Fit a GP with a Matern 3/2 kernel on a return window.

    Parameters
    ----------
    returns : array-like, shape (n,)
        Standardized returns.

    Returns
    -------
    nlml : float
        Negative log marginal likelihood at the optimum.
    params : np.ndarray
        [sigma_f, lengthscale, sigma_n].
    """
    y = np.asarray(returns, dtype=np.float64).ravel()
    X = np.arange(len(y), dtype=np.float64)
    return _fit_base_matern(X, y)


def fit_changepoint_kernel(returns):
    """Fit a GP with the sigmoid-blended changepoint kernel.

    Parameters
    ----------
    returns : array-like, shape (n,)
        Standardized returns.

    Returns
    -------
    nlml : float
        Negative log marginal likelihood at the optimum.
    params : np.ndarray
        [sigma_f1, l1, sigma_f2, l2, c, s, sigma_n].
    """
    y = np.asarray(returns, dtype=np.float64).ravel()
    X = np.arange(len(y), dtype=np.float64)

    # First fit the base to get initialization
    _, base_params = _fit_base_matern(X, y)
    return _fit_changepoint_with_retry(X, y, base_params)





# ---------------------------------------------------------------------------
# Binary Segmentation (offline)
# ---------------------------------------------------------------------------


def binary_segmentation(
    returns: np.ndarray,
    penalty_mult: float = 0.25,
    model: str = "rbf",
) -> list[int]:
    """Offline CPD via Binary Segmentation from the ruptures library.

    The ``penalty_mult`` is a multiplier on a BIC-style scaling::

        penalty = penalty_mult * log(n) * var(returns)

    This makes the penalty scale-invariant -- daily equity returns have
    std ≈ 1% so a raw penalty of 1.0 is effectively "don't split".
    Scaling by the series variance gives a reasonable number of breaks
    across asset classes.

    Parameters
    ----------
    returns : (n,) array of returns.
    penalty_mult : multiplier on the BIC-style penalty term.
    model : ruptures cost model (default ``"l2"``).

    Returns
    -------
    breaks : list of int
        Detection indices.  No continuous score (offline method).
    """
    n = len(returns)
    sigma2 = returns.var()
    pen = penalty_mult * np.log(n) * sigma2
    algo = rpt.Binseg(model=model).fit(returns.reshape(-1, 1))
    breaks = algo.predict(pen=pen)
    # ruptures includes the final index n; drop it
    return [b for b in breaks if b < n]


# ---------------------------------------------------------------------------
# CUSUM (online, combined mean + variance)
# ---------------------------------------------------------------------------


def cusum_combined(
    returns: np.ndarray,
    ref_window: int = 60,
    h_mean: float = 4.0,
    h_var: float = 4.0,
    k_mean: float = 0.5,
    k_var: float = 0.5,
    cooldown: int = 20,
) -> tuple[list[int], np.ndarray]:
    """Online CPD via combined mean + variance CUSUM.

    Maintains two-sided CUSUM statistics for the mean and a one-sided
    statistic for the variance, each referenced against a rolling
    ``ref_window`` estimate of local mean and standard deviation.

    Parameters
    ----------
    returns : (n,) array of returns.
    ref_window : number of observations used to estimate the reference
        mean and standard deviation at each step.
    h_mean : alert threshold for the mean CUSUM statistics.
    h_var : alert threshold for the variance CUSUM statistic.
    k_mean : allowance (slack) parameter for the mean CUSUM.
    k_var : allowance (slack) parameter for the variance CUSUM.
    cooldown : minimum number of observations between consecutive
        detections; statistics are reset after each detection.

    Returns
    -------
    dets : list of int
        Indices at which a changepoint was detected.
    score : (n,) array of float
        Continuous score in [0, 1] via tanh normalisation of the
        dominant statistic (NaN for the first ``ref_window`` steps).
    """
    n = len(returns)
    dets, last = [], -cooldown - 1
    s_pos, s_neg, v_stat = 0.0, 0.0, 0.0
    score = np.full(n, np.nan)

    for t in range(ref_window, n):
        ref = returns[t - ref_window : t]
        mu, sigma = ref.mean(), ref.std(ddof=1)
        if sigma < 1e-12:
            continue
        z = (returns[t] - mu) / sigma

        # Mean CUSUM (two-sided)
        s_pos = max(0.0, s_pos + z - k_mean)
        s_neg = max(0.0, s_neg - z - k_mean)
        # Variance CUSUM
        v_stat = max(0.0, v_stat + (z * z - 1.0) - k_var)

        # Continuous score -- normalise the max statistic by its threshold
        max_mean = max(s_pos, s_neg)
        score[t] = np.tanh(max(max_mean / h_mean, v_stat / h_var))

        trigger = (max_mean > h_mean) or (v_stat > h_var)
        if trigger and (t - last) > cooldown:
            dets.append(t); last = t
            s_pos = s_neg = v_stat = 0.0   # reset

    return dets, score


# ---------------------------------------------------------------------------
# BOCPD (Bayesian Online Changepoint Detection)
# ---------------------------------------------------------------------------


def _log_gaussian_pdf(x: float, mu: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    """Log probability density of a Gaussian evaluated at scalar x.

    Parameters
    ----------
    x : scalar observation.
    mu : (m,) array of means.
    sigma2 : (m,) array of variances (must be positive).

    Returns
    -------
    log_p : (m,) array of log-densities.
    """
    return -0.5 * (np.log(2 * np.pi * sigma2) + (x - mu) ** 2 / sigma2)


def bocpd(
    returns: np.ndarray,
    hazard: float = 1 / 500,
    prior_mu: float = 0.0,
    kappa0: float = 1.0,
    alpha0: float = 1.0,
    beta0: float = 1e-4,
    cooldown: int = 20,
    drop_threshold: int = 30,
    fresh_rl: int = 5,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Bayesian Online Changepoint Detection (BOCPD) with a NIG prior.

    Implements Adams & MacKay (2007) with a Normal-Inverse-Gamma
    (Student-t predictive) observation model.  Detections are triggered
    when the MAP run-length drops by at least ``drop_threshold`` in a
    single step (indicating a reset of the run-length distribution).

    The continuous score is the posterior probability that the current
    run length is at most ``fresh_rl`` observations, i.e.
    ``P(r_t <= fresh_rl)``.

    Parameters
    ----------
    returns : (n,) array of returns.
    hazard : constant hazard rate (prior probability of a changepoint
        at each step).  Default 1/500 corresponds to one changepoint
        expected every 500 observations.
    prior_mu : prior mean for the NIG model.
    kappa0 : prior pseudo-count on the mean.
    alpha0 : prior shape of the inverse-gamma on the variance.
    beta0 : prior scale of the inverse-gamma on the variance.
    cooldown : minimum number of observations between consecutive
        detections.
    drop_threshold : minimum MAP run-length drop (in observations)
        required to register a detection.
    fresh_rl : run-length threshold used for the continuous score;
        ``score[t] = P(r_t <= fresh_rl)``.

    Returns
    -------
    dets : list of int
        Indices at which a changepoint was detected.
    map_run_length : (n,) int array
        MAP run-length estimate at each time step.
    score : (n,) float array
        Continuous changepoint score in [0, 1].
    """
    n = len(returns)
    log_R = np.zeros(1)
    mu_arr    = np.array([prior_mu]);  kappa_arr = np.array([kappa0])
    alpha_arr = np.array([alpha0]);    beta_arr  = np.array([beta0])

    map_rl = np.zeros(n, dtype=int)
    score  = np.zeros(n)
    dets, last = [], -cooldown - 1

    log_haz  = np.log(hazard)
    log_1mh  = np.log(1 - hazard)
    MAX_RL   = 400

    for t in range(n):
        x = returns[t]
        pred_var = beta_arr * (kappa_arr + 1) / (alpha_arr * kappa_arr)
        log_pred = _log_gaussian_pdf(x, mu_arr, pred_var)

        log_growth = log_R + log_pred + log_1mh
        log_cp     = logsumexp(log_R + log_pred + log_haz)
        log_R      = np.concatenate([[log_cp], log_growth])
        log_R     -= logsumexp(log_R)

        mu_new    = (kappa_arr * mu_arr + x) / (kappa_arr + 1)
        kappa_new = kappa_arr + 1
        alpha_new = alpha_arr + 0.5
        beta_new  = beta_arr + 0.5 * kappa_arr * (x - mu_arr) ** 2 / (kappa_arr + 1)
        mu_arr    = np.concatenate([[prior_mu], mu_new])
        kappa_arr = np.concatenate([[kappa0],   kappa_new])
        alpha_arr = np.concatenate([[alpha0],   alpha_new])
        beta_arr  = np.concatenate([[beta0],    beta_new])

        if len(log_R) > MAX_RL:
            log_R     = log_R[:MAX_RL];     log_R -= logsumexp(log_R)
            mu_arr    = mu_arr[:MAX_RL];    kappa_arr = kappa_arr[:MAX_RL]
            alpha_arr = alpha_arr[:MAX_RL]; beta_arr  = beta_arr[:MAX_RL]

        map_rl[t] = int(np.argmax(log_R))
        # Continuous score: P(run length <= fresh_rl)
        score[t]  = float(np.exp(logsumexp(log_R[: fresh_rl + 1])))

        if t > 0:
            drop = map_rl[t - 1] - map_rl[t]
            if drop >= drop_threshold and (t - last) > cooldown:
                dets.append(t); last = t

    return dets, map_rl, score