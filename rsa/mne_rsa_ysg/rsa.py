# encoding: utf-8
"""Methods to compute representational similarity analysis (RSA)."""

import numpy as np
from joblib import Parallel, delayed
from scipy import stats
#import torch

from .folds import create_folds
from .rdm import _ensure_condensed, compute_rdm, compute_rdm_cv
from .searchlight import searchlight

try:
    # Version 1.8.0 and up
    from scipy.stats._stats_py import _kendall_dis
except ImportError:
    from scipy.stats.stats import _kendall_dis

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
def _kendall_tau_a(x, y):
    """Compute Kendall's Tau metric, A-variant.

    Taken from scipy.stats.kendalltau and modified to be the tau-a variant.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.size != y.size:
        raise ValueError(
            "All inputs to `kendalltau` must be of the same size,"
            " found x-size %s and y-size %s" % (x.size, y.size)
        )
    elif not x.size or not y.size:
        return np.nan  # Return NaN if arrays are empty

    def count_rank_tie(ranks):
        cnt = np.bincount(ranks).astype("int64", copy=False)
        cnt = cnt[cnt > 1]
        return (
            (cnt * (cnt - 1) // 2).sum(),
            (cnt * (cnt - 1.0) * (cnt - 2)).sum(),
            (cnt * (cnt - 1.0) * (2 * cnt + 5)).sum(),
        )

    size = x.size
    perm = np.argsort(y)  # sort on y and convert y to dense ranks
    x, y = x[perm], y[perm]
    y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype="intp")

    # stable sort on x and convert x to dense ranks
    perm = np.argsort(x, kind="mergesort")
    x, y = x[perm], y[perm]
    x = np.r_[True, x[1:] != x[:-1]].cumsum(dtype="intp")

    dis = _kendall_dis(x, y)  # discordant pairs

    obs = np.r_[True, (x[1:] != x[:-1]) | (y[1:] != y[:-1]), True]
    cnt = np.diff(np.nonzero(obs)[0]).astype("int64", copy=False)

    ntie = (cnt * (cnt - 1) // 2).sum()  # joint ties
    xtie, x0, x1 = count_rank_tie(x)  # ties in x, stats
    ytie, y0, y1 = count_rank_tie(y)  # ties in y, stats

    tot = (size * (size - 1)) // 2

    if xtie == tot or ytie == tot:
        return np.nan

    # Note that tot = con + dis + (xtie - ntie) + (ytie - ntie) + ntie
    #               = con + dis + xtie + ytie - ntie
    con_minus_dis = tot - xtie - ytie + ntie - 2 * dis
    tau = con_minus_dis / tot
    # Limit range to fix computational errors
    tau = min(1.0, max(-1.0, tau))

    return tau


def _consolidate_masks(masks):
    if type(masks[0]) == slice:
        mask = slice(None)
    else:
        mask = masks[0]
        for m in masks[1:]:
            mask &= m
    return mask


def _partial_correlation(rdm_data, rdm_model, masks=None, type="pearson"):
    """Compute partial Pearson/Spearman correlation."""
    if len(rdm_model) == 1:
        raise ValueError(
            "Need more than one model RDM to use partial " "correlation as metric."
        )
    if type not in ["pearson", "spearman"]:
        raise ValueError("Correlation type must be either 'pearson' or " "'spearman'")

    if masks is not None:
        mask = _consolidate_masks(masks)
        rdm_model = [rdm[mask] for rdm in rdm_model]
        rdm_data = rdm_data[mask]

    X = np.vstack([rdm_data] + rdm_model).T
    if type == "spearman":
        X = np.apply_along_axis(stats.rankdata, 0, X)
    X = X - X.mean(axis=0)
    cov_X_inv = np.linalg.pinv(X.T @ X)
    norm = np.sqrt(np.outer(np.diag(cov_X_inv), np.diag(cov_X_inv)))
    R_partial = cov_X_inv / norm
    return -R_partial[0, 1:]


def rsa_gen(rdm_data_gen, rdm_model, metric="spearman", ignore_nan=False):
    """Generate RSA values between data and model RDMs.

    Will yield RSA scores for each data RDM.

    Parameters
    ----------
    rdm_data_gen : generator of ndarray, shape (n_items, n_items)
        The generator for data RDMs
    rdm_model : ndarray, shape (n_items, n_items) | list of ndarray
        The model RDM, or list of model RDMs.
    metric : str
        The RSA metric to use to compare the RDMs. Valid options are:

        * 'spearman' for Spearman's correlation (the default)
        * 'pearson' for Pearson's correlation
        * 'kendall-tau-a' for Kendall's Tau (alpha variant)
        * 'partial' for partial Pearson correlations
        * 'partial-spearman' for partial Spearman correlations
        * 'regression' for linear regression weights

        Defaults to 'spearman'.
    ignore_nan : bool
        Whether to treat NaN's as missing values and ignore them when computing the
        distance metric. Defaults to ``False``.

        .. versionadded:: 0.8

    Yields
    ------
    rsa_val : float | ndarray, shape (len(rdm_model),)
        For each data RDM, the representational similarity with the model RDM. When
        multiple model RDMs are specified, this will be a 1D array of similarities,
        comparing the data RDM with each model RDM.

    See Also
    --------
    rsa

    """
    if isinstance(rdm_model, list):
        return_array = True
        rdm_model = [_ensure_condensed(rdm, "rdm_model") for rdm in rdm_model]
    else:
        return_array = False
        rdm_model = [_ensure_condensed(rdm_model, "rdm_model")]

    if ignore_nan:
        masks = [~np.isnan(rdm) for rdm in rdm_model]
    else:
        masks = [slice(None)] * len(rdm_model)

    for rdm_data in rdm_data_gen:
        rdm_data = _ensure_condensed(rdm_data, "rdm_data")
        if ignore_nan:
            data_mask = ~np.isnan(rdm_data)
            masks = [m & data_mask for m in masks]
        rsa_vals = _rsa_single_rdm(rdm_data, rdm_model, metric, masks)
        if return_array:
            yield np.asarray(rsa_vals)
        else:
            yield rsa_vals[0]


def _rsa_single_rdm(rdm_data, rdm_model, metric, masks):
    """Compute RSA between a single data RDM and one or more model RDMs."""
    if metric == "spearman":
        rsa_vals = [
            stats.spearmanr(rdm_data[mask], rdm_model_[mask])[0]
            for rdm_model_, mask in zip(rdm_model, masks)
        ]
    elif metric == "pearson":
        rsa_vals = [
            stats.pearsonr(rdm_data[mask], rdm_model_[mask])[0]
            for rdm_model_, mask in zip(rdm_model, masks)
        ]
    elif metric == "kendall-tau-a":
        rsa_vals = [
            _kendall_tau_a(rdm_data[mask], rdm_model_[mask])
            for rdm_model_, mask in zip(rdm_model, masks)
        ]
    elif metric == "partial":
        rsa_vals = _partial_correlation(rdm_data, rdm_model, masks)
    elif metric == "partial-spearman":
        rsa_vals = _partial_correlation(rdm_data, rdm_model, masks, type="spearman")
    elif metric == "regression":
        mask = _consolidate_masks(masks)
        rdm_model = [rdm[mask] for rdm in rdm_model]
        rdm_data = rdm_data[mask]
        X = np.atleast_2d(np.array(rdm_model)).T
        X = X - X.mean(axis=0)
        y = rdm_data - rdm_data.mean()
        weights = np.linalg.lstsq(X, y, rcond=None)[0]

        # 计算预测值和残差
        y_pred = X @ weights
        ss_total = np.sum((y - y.mean()) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)

        # 计算 R-squared
        r_squared = 1 - (ss_residual / ss_total)

        # 返回权重和 R-squared 值
        rsa_vals = (weights, r_squared)




    elif metric == "partial-regression":
        # Consolidate mask
        mask = _consolidate_masks(masks)
        
        # Apply mask to the data
        rdm_model = [rdm[mask] for rdm in rdm_model]
        rdm_data = rdm_data[mask]

        # Convert predictors (X) and output (y) to numpy arrays
        X = np.column_stack(rdm_model)  # Stack the predictors
        y = rdm_data

        # Center y (response variable)
        y = y - np.mean(y)

        # Center X (subtract mean of each column)
        X = X - X.mean(axis=0)

        # Perform full regression with all predictors
        weights, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_pred_full = np.dot(X, weights)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual_full = np.sum((y - y_pred_full) ** 2)
        r_squared_overall = 1 - (ss_residual_full / ss_total)

        # List to store unique contributions (partial R^2 values)
        partial_r_squared = []

        # Loop through each group of predictors
        for k in range(X.shape[1]):
            # Exclude the k-th predictor group
            X_not_k = np.delete(X, k, axis=1)

            # Perform regression without the k-th predictor group
            weights_not_k, _, _, _ = np.linalg.lstsq(X_not_k, y, rcond=None)
            y_pred_not_k = np.dot(X_not_k, weights_not_k)
            ss_residual_not_k = np.sum((y - y_pred_not_k) ** 2)
            r_squared_not_k = 1 - (ss_residual_not_k / ss_total)

            # Calculate the unique contribution of the k-th group
            unique_contribution = r_squared_overall - r_squared_not_k
            partial_r_squared.append(unique_contribution)

        # Return weights, partial R^2 values, and overall R^2
        return weights, partial_r_squared, r_squared_overall



    else:
        raise ValueError(
            "Invalid RSA metric, must be one of: 'spearman', "
            "'pearson', 'partial', 'partial-spearman', "
            "'regression' or 'kendall-tau-a'."
        )
    return rsa_vals


def rsa(
    rdm_data,
    rdm_model,
    metric="spearman",
    ignore_nan=False,
    n_data_rdms=None,
    n_jobs=1,
    verbose=False,
):
    """Perform RSA between data and model RDMs.

    Parameters
    ----------
    rdm_data : ndarray, shape (n_items, n_items) | list | generator
        The data RDM (or list/generator of data RDMs).
    rdm_model : ndarray, shape (n_items, n_items) | list of ndarray
        The model RDM (or list of model RDMs).
    metric : str
        The RSA metric to use to compare the RDMs. Valid options are:

        * 'spearman' for Spearman's correlation (the default)
        * 'pearson' for Pearson's correlation
        * 'kendall-tau-a' for Kendall's Tau (alpha variant)
        * 'partial' for partial Pearson correlations
        * 'partial-spearman' for partial Spearman correlations
        * 'regression' for linear regression weights

        Defaults to 'spearman'.
    ignore_nan : bool
        Whether to treat NaN's as missing values and ignore them when computing
        the distance metric. Defaults to ``False``.

        .. versionadded:: 0.8
    n_data_rdms : int | None
        The number of data RDMs. This is useful when displaying a progress bar, so an
        estimate can be made of the computation time remaining. This information is
        available if ``rdm_data`` is an array or a list, but if it is a generator, this
        information is not available and you may want to set it explicitly.
    n_jobs : int
        The number of processes (=number of CPU cores) to use. Specify -1 to use all
        available cores. Defaults to 1.
    verbose : bool
        Whether to display a progress bar. In order for this to work, you need the tqdm
        python module installed. Defaults to False.

    Returns
    -------
    rsa_val : float | ndarray, shape (len(rdm_data), len(rdm_model))
        Depending on whether one or more data and model RDMs were specified, a single
        similarity value or a 2D array of similarity values for each data RDM versus
        each model RDM.

    See Also
    --------
    rsa_gen

    """
    return_array = False
    if isinstance(rdm_data, list) or hasattr(rdm_data, "__next__"):
        return_array = True
    else:
        rdm_data = [rdm_data]

    if verbose:
        from tqdm import tqdm

        if n_data_rdms is not None:
            total = n_data_rdms
        elif hasattr(rdm_data, "__len__"):
            total = len(rdm_data)
        else:
            total = None
        rdm_data = tqdm(rdm_data, total=total, unit="RDM")

    if n_jobs == 1:
        rsa_vals = list(rsa_gen(rdm_data, rdm_model, metric, ignore_nan))
    else:

        def process_single_rdm(rdm):
            return next(rsa_gen([rdm], rdm_model, metric, ignore_nan))

        rsa_vals = Parallel(n_jobs)(
            delayed(process_single_rdm)(rdm) for rdm in rdm_data
        )
    if return_array:
        return np.asarray(rsa_vals)
    else:
        return rsa_vals[0]


def rsa_array(
    X,
    rdm_model,
    patches=None,
    data_rdm_metric="correlation",
    data_rdm_params=dict(),
    rsa_metric="spearman",
    ignore_nan=False,
    y=None,
    n_folds=1,
    n_jobs=1,
    verbose=False,
):
    """Perform RSA on an array of data, possibly in a searchlight pattern."""

    if patches is None:
        patches = searchlight(X.shape)  # One big searchlight patch

    # Create folds for cross-validated RDM metrics
    X = create_folds(X, y, n_folds)
    # The data is now folds x items x n_series x n_times

    if isinstance(rdm_model, list):
        rdm_model = [_ensure_condensed(rdm, "rdm_model") for rdm in rdm_model]
    else:
        rdm_model = [_ensure_condensed(rdm_model, "rdm_model")]

    if ignore_nan:
        masks = [~np.isnan(rdm) for rdm in rdm_model]
    else:
        masks = [slice(None)] * len(rdm_model)

    if verbose:
        from tqdm import tqdm

        shape = getattr(patches, "shape", (-1,))
        patches = tqdm(patches, unit="patch")
        try:
            setattr(patches, "shape", shape)
        except AttributeError:
            pass

    def rsa_single_patch(patch):
        """Compute RSA for a single searchlight patch."""
        if len(X) == 1:  # Check number of folds
            # No cross-validation
            rdm_data = compute_rdm(X[0][patch], data_rdm_metric, **data_rdm_params)
        else:
            # Use cross-validation
            rdm_data = compute_rdm_cv(
                X[(slice(None),) + patch], data_rdm_metric, **data_rdm_params
            )
        if ignore_nan:
            data_mask = ~np.isnan(rdm_data)
            patch_masks = [m & data_mask for m in masks]
        else:
            patch_masks = masks
        
        rsa_result = _rsa_single_rdm(rdm_data, rdm_model, rsa_metric, patch_masks)

        if rsa_metric == "regression":
            weights, r_squared = rsa_result
            return weights, r_squared  # 返回权重和r方
        
        return rsa_result

    # Call RSA multiple times in parallel for each searchlight patch
    data = Parallel(n_jobs=n_jobs)(
        delayed(rsa_single_patch)(patch) for patch in patches
    )

    # 处理返回的权重和r方
    if rsa_metric == "regression":
        weights, r_squared = zip(*data)
        print(f"patches shape: {getattr(patches, 'shape', None)}")
        print(f"weights shape: {np.array(weights).shape}")
        print(f"r_squared shape: {np.array(r_squared).shape}")
        #weights, r_squared = zip(*data)
        weights = np.array(weights)  # shape: (81160, 2)
        r_squared = np.array(r_squared)  # shape: (81160,)
        return weights, r_squared
    
    if rsa_metric == "partial-regression":
        results_pr = []
        weights, partial_r_squared, r_squared_overall = zip(*data)
        #print(f"patches shape: {getattr(patches, 'shape', None)}")
        #print(f"weights shape: {np.array(weights).shape}") # shape: (81160, 2)
        #print(f"partial_r_squared shape: {np.array(partial_r_squared).shape}")# shape: (81160,)
        #print(f"r_squared shape: {np.array(r_squared_overall).shape}")# shape: (81160,)
        #weights, r_squared = zip(*data)
        w_arr_split = np.split(np.array(weights),np.array(weights).shape[1], axis=1)
        for item1 in w_arr_split:
            #print(item1.shape)
            results_pr.append(item1)
        arr_split = np.split(np.array(partial_r_squared), np.array(partial_r_squared).shape[1], axis=1)
        for itemm in arr_split:
            #print(itemm.shape)
            results_pr.append(itemm)
        all_r_squared = np.array(r_squared_overall)  # shape: (81160,)
        results_pr.append(all_r_squared)
        return results_pr
    
    # 如果不是regression，返回单一值
    dims = getattr(patches, "shape", (-1,))
    if len(rdm_model) > 1:
        dims = dims + (len(rdm_model),)

    return np.array(data).reshape(dims)

