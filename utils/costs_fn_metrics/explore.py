import itertools
import os
from typing import Iterable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import ott.geometry.costs as costs
import pandas as pd
import tqdm
from numpy.typing import ArrayLike

from utils.costs_fn_metrics.matching import rowwise_take_top_k, sinkhorn_matching
from utils.costs_fn_metrics.metrics import (
    discrete_wasserstein_distance,
    rowwise_correlation,
    similarity_top_k,
)


def get_batch(
    data: ArrayLike,
    label: Optional[ArrayLike] = None,
    batch_size: int = 64,
    batch_idx: Optional[ArrayLike] = None,
    key=None,
):
    N = data.shape[0]
    if batch_idx is None:
        batch_idx = jax.random.randint(key, (batch_size,), 0, N)

    if label is not None:
        return data[batch_idx], label[batch_idx], batch_idx
    return data[batch_idx], batch_idx


def resample(
    matrix: jax.Array,
    batchX: jax.Array,
    batchY: jax.Array,
    key: jr.KeyArray,
    batch_labelX: Optional[jax.Array] = None,
    batch_labelY: Optional[jax.Array] = None,
    batch_size: int = 256,
    logprob: bool = True,
) -> Tuple[jax.Array, jax.Array]:
    if logprob:
        transition_matrix = jnp.log(matrix.flatten())
    else:
        transition_matrix = matrix.flatten()

    indeces = jax.random.categorical(key, transition_matrix, shape=[batch_size])
    resampled_indeces_source = indeces // batch_size
    resampled_indeces_target = indeces % batch_size

    if batch_labelX is None:
        return (
            batchX[resampled_indeces_source],
            batchY[resampled_indeces_target],
        )
    return (
        batchX[resampled_indeces_source],
        batchY[resampled_indeces_target],
        batch_labelX[resampled_indeces_source],
        batch_labelY[resampled_indeces_target],
    )


def single_cost_fn_metrics(
    matrix: ArrayLike,
    labelX: ArrayLike,
    labelY: ArrayLike,
    key: jr.KeyArray,
    batch_size: int,
):
    metrics = dict()
    for top_k in [1, 3, 5, 16, 32]:
        top_k_labelsY = rowwise_take_top_k(matrix, top_k=top_k, labels=labelY)
        # Assume one hot label encoding for X and Y is comparable, otherwise you must convert between them
        sim_top_k = similarity_top_k(labelX, top_k_labels=top_k_labelsY)
        metrics[f"top_{top_k}"] = sim_top_k

    for bottom_k in [1, 3, 5, 16, 32]:
        bottom_k_labelsY = rowwise_take_top_k(-matrix, top_k=bottom_k, labels=labelY)
        sim_bottom_k = similarity_top_k(labelX, top_k_labels=bottom_k_labelsY)

        metrics[f"bottom_{bottom_k}"] = sim_bottom_k

    sample_labelX, sample_labelY = resample(
        matrix=matrix,
        batchX=labelX,
        batchY=labelY,
        key=key,
        batch_size=batch_size,
    )
    sim_sample = similarity_top_k(
        sample_labelX,
        top_k_labels=sample_labelY[:, None, ...],
    )

    metrics[f"sample"] = sim_sample

    return metrics


def matrix_comparison_metrics(
    matrix0: ArrayLike,
    matrix1: ArrayLike,
):
    row_correlation = rowwise_correlation(matrix0, matrix1)
    col_correlation = rowwise_correlation(jnp.transpose(matrix0), np.transpose(matrix1))
    metrics = dict(
        byrow_mean=jnp.mean(row_correlation),
        byrow_std=jnp.std(row_correlation),
        byrow_max=jnp.max(row_correlation),
        byrow_min=jnp.min(row_correlation),
        bycol_mean=jnp.mean(col_correlation),
        bycol_std=jnp.std(col_correlation),
        bycol_max=jnp.max(col_correlation),
        bycol_min=jnp.min(col_correlation),
    )

    # ! SLOW & MEMORY CONSUMING. Limit when it gets calculated (batch_size <= 50)
    if np.prod(matrix0.shape) <= 2500:
        wasserstein0_distance = discrete_wasserstein_distance(
            matrix0.flatten(), matrix1.flatten()
        )
        metrics.update(dict(W0=wasserstein0_distance))

    return metrics


def comparison_plot(
    X: ArrayLike,
    compX: ArrayLike,
    k: Optional[int] = None,
    limit_elements_to_display: int = 32,
):
    B = min(limit_elements_to_display, X.shape[0])
    if k is None:
        k = compX.shape[1]

    fig, axes = plt.subplots(B, 1 + k, figsize=(k, B))
    for b in range(B):
        axes[b, 0].imshow(X[b].transpose(1, 2, 0), cmap="gray")
        axes[b, 0].axis("off")

        for ki in range(k):
            axes[b, 1 + ki].imshow(compX[b, ki].transpose(1, 2, 0), cmap="gray")
            axes[b, 1 + ki].axis("off")

    return (
        fig,
        axes,
    )


def sample_plot(
    X: ArrayLike,
    Y: ArrayLike,
    nrow: int = 8,
    ncol: int = 8,
):
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol, nrow))

    for i in range(nrow):
        for j in range(ncol):
            if not (j % 2):
                axes[i, j].imshow(
                    X[i * (ncol // 2) + j // 2].transpose(1, 2, 0), cmap="gray"
                )
                if not i:
                    axes[i, j].set_title("X")
            else:
                axes[i, j].imshow(
                    Y[i * (ncol // 2) + j // 2].transpose(1, 2, 0), cmap="gray"
                )
                if not i:
                    axes[i, j].set_title("Y")

            axes[i, j].axis("off")

    return fig, axes


def explore_cost_fn(
    X: np.ndarray,
    labelX: np.ndarray,
    cost_fn: Union[costs.CostFn, Iterable[costs.CostFn]],
    sinkhorn_matching_kwargs: Optional[dict] = None,
    Y: Optional[np.ndarray] = None,
    labelY: Optional[np.ndarray] = None,
    nbatches: int = 1,
    batch_size: int = 32,
    summarize: bool = True,
    save_folder: Optional[str] = None,
    seed: int = 0,
    overwrite: bool = False,
):
    if sinkhorn_matching_kwargs is None:
        sinkhorn_matching_kwargs = dict()

    if isinstance(cost_fn, costs.CostFn):
        cost_fn = [cost_fn]
    cost_fn = {cost_fni.__class__.__name__: cost_fni for cost_fni in cost_fn}

    if Y is None:  # Perform self comparison
        Y = X
        labelY = labelX

    # ---
    key = jax.random.key(seed)

    metrics = []
    comparison_metrics = []
    batch_idxs = dict(X=[], Y=[])
    for _ in tqdm.tqdm(range(nbatches)):
        key, key_batchX, key_batchY = jax.random.split(key, 3)

        # Obtain batch and corresponding sinkhorn matrix
        batchX, batch_labelX, batch_idxX = get_batch(
            X,
            labelX,
            batch_size=batch_size,
            key=key_batchX,
        )
        batch_idxs["X"].append(batch_idxX)

        batchY, batch_labelY, batch_idxY = get_batch(
            Y,
            labelY,
            batch_size=batch_size,
            key=key_batchY,
        )
        batch_idxs["Y"].append(batch_idxY)

        batch_matrixs = dict()
        batch_metrics = dict()
        for cost_fn_namei, cost_fni in cost_fn.items():
            key, key_single_cost_fn_metrics = jax.random.split(key, 2)

            matrixi = sinkhorn_matching(
                X=batchX,
                Y=batchY,
                cost_fn=cost_fni,
                **sinkhorn_matching_kwargs,
            )

            batch_metricsi = single_cost_fn_metrics(
                matrixi,
                batch_labelX,
                batch_labelY,
                key=key_single_cost_fn_metrics,
                batch_size=batch_size,
            )

            batch_metrics[cost_fn_namei] = batch_metricsi
            batch_matrixs[cost_fn_namei] = matrixi

        # Perform pairwise comparison
        batch_comparison_metrics = dict()
        for cost_fn_namei, cost_fn_namej in itertools.combinations(cost_fn.keys(), 2):
            matrixi = batch_matrixs[cost_fn_namei]
            matrixj = batch_matrixs[cost_fn_namej]
            batch_comparison_metrics[f"{cost_fn_namei}_{cost_fn_namej}"] = (
                matrix_comparison_metrics(matrixi, matrixj)
            )

        comparison_metrics.append(batch_comparison_metrics)
        metrics.append(batch_metrics)

    # Format metrics & Comparison metrics
    # ----
    # Swap batches and cost_fn_name keys
    metrics = {
        cost_fn_namei: [metrics[i][cost_fn_namei] for i in range(len(metrics))]
        for cost_fn_namei in metrics[0]
    }
    # Concatenate fieldwise all the array corresponding to the metric for a cost_fn
    metrics = {
        cost_fn_namei: jax.tree_map(lambda *arr: jnp.stack(arr), *cost_fn_metricsi)
        for cost_fn_namei, cost_fn_metricsi in metrics.items()
    }

    metrics = {
        cost_fn_namei: pd.DataFrame.from_dict(cost_fn_metricsi)
        for cost_fn_namei, cost_fn_metricsi in metrics.items()
    }

    # ---
    # Repeat same operation for comparison metrics
    comparison_metrics = {
        cost_fn_pairname: [
            comparison_metrics[i][cost_fn_pairname]
            for i in range(len(comparison_metrics))
        ]
        for cost_fn_pairname in comparison_metrics[0]
    }

    comparison_metrics = {
        cost_fn_pairname: jax.tree_map(
            lambda *arr: jnp.stack(arr), *cost_fn_pairname_metrics
        )
        for cost_fn_pairname, cost_fn_pairname_metrics in comparison_metrics.items()
    }
    comparison_metrics = {
        cost_fn_pairname: pd.DataFrame.from_dict(cost_fn_pairname_metrics)
        for cost_fn_pairname, cost_fn_pairname_metrics in comparison_metrics.items()
    }

    # Figures.
    if save_folder:
        os.makedirs(save_folder, exist_ok=overwrite)
        # Figures. Random batch matching (Same for all costs)
        key, key_random_batch = jax.random.split(key, 2)
        random_batch_idx = jax.random.randint(key_random_batch, (), 0, nbatches)
        batch_idxX = batch_idxs["X"][random_batch_idx]
        batch_idxY = batch_idxs["Y"][random_batch_idx]
        batchX = X[batch_idxX]
        batchY = Y[batch_idxY]

        for cost_fn_namei, cost_fni in cost_fn.items():

            # Figures. Top and Bottom in Random batch matching
            cost_fn_folderi = os.path.join(save_folder, cost_fn_namei)
            os.makedirs(cost_fn_folderi, exist_ok=overwrite)
            # Figures. Top
            matrixi = sinkhorn_matching(
                X=batchX,
                Y=batchY,
                cost_fn=cost_fni,
                **sinkhorn_matching_kwargs,
            )

            top_k_elements = rowwise_take_top_k(matrixi, top_k=16, labels=batchY)
            fig, ax = comparison_plot(
                batchX,
                compX=top_k_elements,
                limit_elements_to_display=32,
            )
            fig.tight_layout()
            fig.savefig(os.path.join(cost_fn_folderi, "top_random_batch.jpg"))

            # Figures. Bottom
            bottom_k_elements = rowwise_take_top_k(-matrixi, top_k=16, labels=batchY)
            fig, ax = comparison_plot(
                batchX,
                compX=bottom_k_elements,
                limit_elements_to_display=32,
            )
            fig.tight_layout()
            fig.savefig(os.path.join(cost_fn_folderi, "bottom_random_batch.jpg"))

            # Figures. Sample
            key, key_sample = jax.random.split(key, 2)
            sampleX, sampleY = resample(
                matrix=matrixi,
                batchX=batchX,
                batchY=batchY,
                key=key_sample,
                batch_size=batch_size,
            )

            fig, ax = sample_plot(X=sampleX, Y=sampleY, nrow=16, ncol=16)
            fig.tight_layout()
            fig.savefig(os.path.join(cost_fn_folderi, "sample_random_batch.jpg"))

            # Figures. Best and Worse Top and bottom k matching
            for metric in ["top_5", "bottom_5"]:
                metric_field = metrics[cost_fn_namei][metric]
                # Figures. Best
                batch_idxX = batch_idxs["X"][np.argmax(metric_field)]
                batch_idxY = batch_idxs["Y"][np.argmax(metric_field)]
                batchX = X[batch_idxX]
                batchY = Y[batch_idxY]

                matrixi = sinkhorn_matching(
                    X=batchX,
                    Y=batchY,
                    cost_fn=cost_fni,
                    **sinkhorn_matching_kwargs,
                )
                # If bottom, we consider the lowest ones
                if metric == "bottom_5":
                    matrixi = -matrixi

                top_k_elements = rowwise_take_top_k(matrixi, top_k=5, labels=batchY)
                fig, ax = comparison_plot(
                    batchX,
                    compX=top_k_elements,
                    limit_elements_to_display=32,
                )

                # fig.suptitle(f"Best {metric}={metric_value:.2f}")
                fig.tight_layout()
                fig.savefig(os.path.join(cost_fn_folderi, f"best_{metric}.jpg"))

                # Figures. Bottom
                batch_idxX = batch_idxs["X"][np.argmin(metric_field)]
                batch_idxY = batch_idxs["Y"][np.argmin(metric_field)]
                batchX = X[batch_idxX]
                batchY = Y[batch_idxY]

                matrixi = sinkhorn_matching(
                    X=batchX,
                    Y=batchY,
                    cost_fn=cost_fni,
                    **sinkhorn_matching_kwargs,
                )

                top_k_elements = rowwise_take_top_k(matrixi, top_k=5, labels=batchY)
                fig, ax = comparison_plot(
                    batchX,
                    compX=top_k_elements,
                    limit_elements_to_display=32,
                )
                # fig.suptitle(f"Worse {metric}={metric_value:.2f}")
                fig.tight_layout()
                fig.savefig(os.path.join(cost_fn_folderi, f"worse_{metric}.jpg"))

    # Metrics Summarization
    if summarize:
        metrics = {
            cost_fn_namei: cost_fn_metricsi.describe().rename(dict(count="nbatches"))
            for cost_fn_namei, cost_fn_metricsi in metrics.items()
        }
        comparison_metrics = {
            cost_fn_pairname: cost_fn_pairname_metrics.describe().rename(
                dict(count="nbatches")
            )
            for cost_fn_pairname, cost_fn_pairname_metrics in comparison_metrics.items()
        }

    if save_folder:
        for cost_fn_namei, cost_fni in cost_fn.items():
            cost_fn_folderi = os.path.join(save_folder, cost_fn_namei)
            metrics[cost_fn_namei].to_csv(os.path.join(cost_fn_folderi, "metrics.csv"))

        if comparison_metrics:
            comparison_fn_folderi = os.path.join(save_folder, "comparison")
            os.makedirs(comparison_fn_folderi, exist_ok=overwrite)
            for comparison_pair_name, comparison_pair_df in comparison_metrics.items():
                # comparison_fn_folderi = os.path.join(save_folder, comparison_pair_name)
                # TODO: The folder will be creating when creating the figures for comparison if done
                # TODO: For now, save into a "comparison folder"

                # os.makedirs(comparison_fn_folderi, exist_ok=overwrite)
                comparison_pair_df.to_csv(
                    os.path.join(comparison_fn_folderi, f"{comparison_pair_name}.csv")
                )

    if len(metrics) == 1:
        metrics = next(iter(metrics.values()))

    return metrics, comparison_metrics
