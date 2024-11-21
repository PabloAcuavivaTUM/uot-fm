import functools as ft
import logging
import warnings
from typing import Callable, Optional

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import ml_collections
import numpy as np
import scipy
import tensorflow as tf
import wandb
from tqdm import tqdm
from PIL import Image

from models import inception

from .miscellaneous import EasyDict, generate_wb_image, jx_device_put


class MetricComputer:
    """
    Class to compute metrics for evaluation.
    """

    def __init__(
        self,
        config: ml_collections.ConfigDict,
        shard: jax.sharding.Sharding,
        eval_ds: tf.data.Dataset,
        sample_fn: Callable,
        vae_decode_fn: Optional[Callable] = None,
        vae_encode_fn: Optional[Callable] = None,
        is_genot: bool = False,
    ):
        # load pretrained inceptionv3 model
        rng = jax.random.PRNGKey(0)
        model = inception.InceptionV3(pretrained=True)
        self.params = model.init(rng, jnp.ones((1, 299, 299, 3)))
        self.apply_fn = jax.jit(ft.partial(model.apply, train=False))
        # get training config
        self.eval_labelwise = config.eval.labelwise
        self.task = config.task
        if self.task == "translation":
            self.num_eval_samples = eval_ds.length
        else:
            self.num_eval_samples = config.eval.eval_samples
        self.batch_size = config.training.batch_size
        self.return_samples = config.eval.save_samples
        if self.return_samples:
            self.num_save_samples = config.eval.num_save_samples
        self.dataset = eval_ds
        self.repeat = 3 if config.data.shape[0] == 1 else 1
        self.input_shape = config.model.input_shape
        self.sample_fn = sample_fn
        self.enable_fid = config.eval.enable_fid
        self.enable_mse = config.eval.enable_mse
        self.enable_path_lengths = config.eval.enable_path_lengths
        if self.enable_mse:
            self.mse_fn = jax.jit(lambda x, y: jnp.mean((x - y) ** 2))
        if self.enable_path_lengths:
            self.rmse_fn = jax.jit(lambda x, y: jnp.mean(jnp.sqrt((x - y) ** 2)))
        self.shard = shard
        self.use_vae = config.model.use_vae
        if self.use_vae:
            assert vae_decode_fn is not None and vae_encode_fn is not None
            self.vae_decode_fn = vae_decode_fn
            self.vae_encode_fn = vae_encode_fn
        if self.enable_fid:
            # get statistics for real data
            stats_file_name = config.data.precomputed_stats_file
            precomputed_stats = np.load(f"./assets/stats/{stats_file_name}.npz")
            self.mu_real, self.sigma_real = (
                precomputed_stats["mu"],
                precomputed_stats["sigma"],
            )
            if self.eval_labelwise:
                self.eval_labels = jnp.array(config.data.eval_labels)
                self.mus_real, self.sigmas_real = [], []
                for label in self.eval_labels:
                    precomputed_stats = np.load(
                        f"./assets/stats/{stats_file_name}_{label}.npz"
                    )
                    self.mus_real.append(precomputed_stats["mu"])
                    self.sigmas_real.append(precomputed_stats["sigma"])

        self.is_genot = is_genot

    def compute_metrics(self, model: eqx.Module, key: jr.KeyArray):
        """
        Compute metrics for evaluation.

        Args:
            model: model to evaluate
            key: jax random key

        Returns:
            eval_dict: dictionary with evaluation metrics and samples for wandb logging
        """
        eval_dict = {}
        if self.eval_labelwise:
            labels_indices = [[] for _ in range(self.eval_labels.shape[0])]
        samples = None
        inputs = None
        mses = []
        path_lengths = []
        inception_acts = []
        nfes = []
        # create vmap functions
        partial_sample_fn = ft.partial(self.sample_fn, model)
        # compute metrics batch-wise
        eval_num_iter = max(self.num_eval_samples // self.batch_size, 1)
        if self.task == "translation":
            loader = iter(self.dataset)
        for _ in tqdm(range(eval_num_iter)):
            if self.task == "translation":
                src_batch = next(loader)
                # padding for last batch if necessary
                pad_size = self.batch_size - src_batch.data.shape[0]
                if pad_size > 0:
                    # TODO (IF using labels for something in the model): Here probably need to pad the whole thing. Also labels/embeddings...
                    src_batch["data"] = jnp.pad(
                        src_batch.data, ((0, pad_size), (0, 0), (0, 0), (0, 0))
                    )
            else:
                sample_key, key = jr.split(key, 2)
                pad_size = 0
                src_batch = EasyDict(
                    data=jr.normal(sample_key, [self.batch_size, *self.input_shape]),
                    label=None,
                )

            src_batch = jx_device_put(src_batch, self.shard)

            if inputs is None:
                if self.use_vae:
                    inputs = self.vae_decode_fn(src_batch.data) * 0.5 + 0.5
                else:
                    inputs = src_batch.data * 0.5 + 0.5
            elif inputs.shape[0] < 2400:
                if self.use_vae:
                    inputs = jnp.concatenate(
                        [
                            inputs,
                            self.vae_decode_fn(src_batch.data)[
                                : int(2400 - inputs.shape[0])
                            ]
                            * 0.5
                            + 0.5,
                        ]
                    )
                else:
                    inputs = jnp.concatenate(
                        [
                            inputs,
                            src_batch.data[: int(2400 - inputs.shape[0])] * 0.5 + 0.5,
                        ]
                    )

            # sample from model
            if self.is_genot:
                batch_size = src_batch.data.shape[0]
                ode_key = jr.split(key, batch_size)
            else:
                ode_key = None

            sample_batch, nfe = jax.vmap(partial_sample_fn)(src_batch, ode_key)

            nfes.append(nfe)
            if self.enable_path_lengths:
                # compute euclidean distance between samples and inputs
                if pad_size > 0:
                    path_lengths.append(
                        self.rmse_fn(
                            src_batch.data[:-pad_size], sample_batch[:-pad_size]
                        )
                    )
                else:
                    path_lengths.append(self.rmse_fn(src_batch.data, sample_batch))
            if self.use_vae:
                sample_batch = jx_device_put(sample_batch, self.shard)
                sample_batch = self.vae_decode_fn(sample_batch)
            sample_batch = jnp.clip(sample_batch, -1.0, 1.0)
            if self.enable_fid:
                inception_act = self.compute_inception_acts(sample_batch)
                inception_acts.append(inception_act)
            if (
                pad_size > 0
            ):  # TODO: What is the function of padding src_batc['data']? It doesn't seem to be used
                src_batch["data"] = src_batch.data[:-pad_size]
                sample_batch = sample_batch[:-pad_size]
            # safe samples and compute inception activation
            if samples is None:
                samples = sample_batch * 0.5 + 0.5
            elif samples.shape[0] < 2400:
                samples = jnp.concatenate(
                    [samples, sample_batch[: int(2400 - samples.shape[0])] * 0.5 + 0.5]
                )
            if self.eval_labelwise:
                for idx, label in enumerate(self.eval_labels):
                    if label == 201:
                        labels_indices[idx].append((src_batch.label[:, 20] == -1))
                    else:
                        labels_indices[idx].append((src_batch.label[:, label] == 1.0))

        eval_dict["nfe"] = jnp.mean(jnp.hstack(nfes))
        if self.enable_mse:
            eval_dict["mse"] = jnp.mean(jnp.hstack(mses))
        if self.enable_path_lengths:
            eval_dict["path_lengths"] = jnp.mean(jnp.hstack(path_lengths)) * 127.5
            eval_dict["path_lengths_std"] = jnp.std(jnp.hstack(path_lengths)) * 127.5

        # compute fid
        if self.enable_fid:
            inception_acts = jnp.concatenate(inception_acts, axis=0)
            if pad_size > 0:
                inception_acts = inception_acts[:-pad_size]
            mu = jnp.mean(inception_acts, axis=0)
            sigma = jnp.cov(inception_acts, rowvar=False)
            eval_dict["fid"] = self.compute_fid(
                self.mu_real, self.sigma_real, mu, sigma
            )
        if self.return_samples:
            # save image grid loggable to wandb
            if self.task == "generation":
                wb_image = generate_wb_image(
                    samples=samples, num_samples=self.num_save_samples
                )
            else:
                wb_image = generate_wb_image(
                    samples=samples, inputs=inputs, num_samples=self.num_save_samples
                )
            eval_dict["samples"] = wb_image

        if self.eval_labelwise:
            # compute fid labelwise
            fid_scores = []
            for idx, label in enumerate(self.eval_labels):
                label_indices = jnp.concatenate(labels_indices[idx], axis=0)
                if self.enable_fid:
                    inception_act_label = inception_acts[label_indices]
                    mu = jnp.mean(inception_act_label, axis=0)
                    sigma = jnp.cov(inception_act_label, rowvar=False)
                    fid_score = self.compute_fid(
                        self.mus_real[idx], self.sigmas_real[idx], mu, sigma
                    )
                    eval_dict[f"fid_{label}"] = fid_score
                    fid_scores.append(fid_score)
                if (
                    self.return_samples
                ):  # ! Notice here we always assume we are in the translation setting for now
                    plot_indices = label_indices[
                        :2400
                    ]  # ? Is 2400 just hardcoded for any reason or just to get enough images?
                    wb_image = generate_wb_image(
                        samples=samples[plot_indices],
                        inputs=inputs[plot_indices],
                        num_samples=self.num_save_samples,
                    )
                    eval_dict[f"samples_{label}"] = wb_image

            eval_dict["fid_average"] = jnp.mean(jnp.hstack(fid_scores))
        return eval_dict

    def compute_inception_acts(self, image_batch: jax.Array) -> jax.Array:
        """
        Compute inception activations for a batch of images.
        """
        inception_input = einops.repeat(
            image_batch, "b c h w -> b h w (c repeat)", repeat=self.repeat
        )
        inception_input = jax.image.resize(
            inception_input,
            shape=[image_batch.shape[0], 299, 299, 3],
            method="bilinear",
            antialias=True,
        )
        inception_output = self.apply_fn(
            self.params, jax.lax.stop_gradient(inception_input)
        )
        return inception_output.squeeze(axis=1).squeeze(axis=1)

    @staticmethod
    def compute_fid(
        mu_real: np.ndarray,
        sigma_real: np.ndarray,
        mu_gen: np.ndarray,
        sigma_gen: np.ndarray,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """
        Compute Frechet Inception Distance (FID) between two distributions.
        """
        # compute statistics
        mu_gen = np.atleast_1d(mu_gen)
        mu_real = np.atleast_1d(mu_real)
        sigma_gen = np.atleast_1d(sigma_gen)
        sigma_real = np.atleast_1d(sigma_real)

        assert (
            mu_gen.shape == mu_real.shape
        ), f"Shapes {mu_gen.shape} != {mu_real.shape}"
        assert (
            sigma_gen.shape == sigma_real.shape
        ), f"Shapes {sigma_gen.shape} != {sigma_real.shape}"

        diff = mu_real - mu_gen
        covmean, _ = scipy.linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)

        if not np.isfinite(covmean).all():
            warnings.warn(
                (
                    f"fid calculation produces singular product; "
                    "adding {eps} to diagonal of cov estimates"
                )
            )
            offset = np.eye(sigma_real.shape[0]) * eps
            covmean = scipy.linalg.sqrtm((sigma_real + offset).dot(sigma_gen + offset))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-2):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component {m}")
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        return (
            diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_gen) - 2 * tr_covmean
        )


####
# IN PROGRESS: Cell metric computer (CLEAN UP)
import io
import pandas as pd
import math 
from typing import Dict 
import seaborn as sns  
from typing import Optional, List


import matplotlib.pyplot as plt
from .cell_fns.cell_metric_fns import (
    calculate_same_class_perc,
    estimate_precision,
    estimate_recall,
)


def np_to_dataframe(array: np.ndarray, columns: Optional[List[str]] = None):
    if len(array.shape) != 2:
        raise ValueError(
            f"Only squared arrays (n,m) can be converted to dataframes! array shape {array.shape}"
        )
    columns = columns or []
    required_columns = array.shape[1]
    columns += [f"Feature_{i}" for i in range(len(columns), required_columns)]

    return pd.DataFrame(data=array, columns=columns)
    

def jnp_safe_concat(x: Optional[jax.Array], y: jax.Array):
    if x is None:
        return y
    return jnp.concatenate([x, y])

def np_safe_concat(x: Optional[np.ndarray], y: np.ndarray):
    if x is None:
        return y
    return np.concatenate([x, y])


def easy_pad(easy_dict: EasyDict, pad_size: int):
    return EasyDict(
        **{
            k: jnp.pad(
                v, [(0, pad_size)] + [(0, 0) for _ in v.shape[1:]],
            )
            for k, v in easy_dict.items()
        }
    )


def easy_unpad(easy_dict : EasyDict, pad_size : int):
    return EasyDict(**{k: v[:-pad_size] for k,v in easy_dict.items()})


###
# Color Constants
helmholtz_primary = (105 / 255, 0 / 255, 95 / 255)
helmholtz_secondary = (255 / 255, 80 / 255, 110 / 255)

tum_primary = (48 / 255, 112 / 255, 179 / 255)
tum_secondary = (94 / 255, 148 / 255, 212 / 255)
tum_orange = (227 / 255, 114 / 255, 34 / 255)
tum_green = (162 / 255, 173 / 255, 0 / 255)
tum_red = (217 / 255, 81 / 255, 23 / 255)
tum_gray = (153 / 255, 153 / 255, 153 / 255)

tum_colors = [tum_primary,
              tum_green,
              tum_red,
              tum_secondary,
              tum_gray,
              tum_orange,
              helmholtz_primary,
              helmholtz_secondary,
              ]

#### 
class CellMetricComputer:
    """
    Class to compute metrics for evaluation.
    """

    def __init__(
        self,
        config: ml_collections.ConfigDict,
        shard: jax.sharding.Sharding,
        eval_src_ds: tf.data.Dataset,
        eval_tgt_ds: tf.data.Dataset,
        auxiliary_data_prep: dict,
        sample_fn: Callable,
        vae_decode_fn: Optional[Callable] = None,
        vae_encode_fn: Optional[Callable] = None,
        is_genot: bool = False,
    ):
        # get training config
        self.num_eval_samples = eval_src_ds.length
        self.batch_size = config.training.batch_size

        self.return_samples = config.eval.save_samples
        if self.return_samples:
            self.num_save_samples = config.eval.num_save_samples

        self.dataset = eval_src_ds
        self.target_dataset = eval_tgt_ds
        self.input_shape = config.model.input_shape
        self.sample_fn = sample_fn
        self.enable_mse = config.eval.enable_mse
        self.enable_path_lengths = config.eval.enable_path_lengths
        if self.enable_mse:
            self.mse_fn = jax.jit(lambda x, y: jnp.mean((x - y) ** 2))
        if self.enable_path_lengths:
            self.rmse_fn = jax.jit(lambda x, y: jnp.mean(jnp.sqrt((x - y) ** 2)))
        self.shard = shard
        self.use_vae = config.model.use_vae

        if self.use_vae:
            assert vae_decode_fn is not None and vae_encode_fn is not None
            self.vae_decode_fn = vae_decode_fn
            self.vae_encode_fn = vae_encode_fn

        self.is_genot = is_genot

        self.auxiliary_data_prep = auxiliary_data_prep

        ###
        # Extract embeddings for metric plotting                
        self.additional_embeddings = dict()
        self.cell_embeddings_metrics = config.eval.cell_embeddings_metrics
        self.cell_embeddings_histograms = config.eval.cell_embeddings_histograms
        additional_embedding = config.data.additional_embedding

        for name, dataset in dict(
            dataset=self.dataset, target_dataset=self.target_dataset
        ).items():
            additional_embedding = {embedding: [] for embedding in additional_embedding}
            eval_num_iter = dataset.length // self.batch_size + 1
            loader = iter(dataset)
            for _ in tqdm(range(eval_num_iter)):
                batch = next(loader)
                for embedding, embedding_value in additional_embedding.items():
                    embedding_value.append(np.asarray(batch[embedding]))

            additional_embedding = {
                embedding: np.concatenate(embedding_value)
                for embedding, embedding_value in additional_embedding.items()
            }

            self.additional_embeddings[name] = additional_embedding

    def compute_metrics(self, model: eqx.Module, key: jr.KeyArray):
        """
        Compute metrics for evaluation on cell data.

        Args:
            model: model to evaluate
            key: jax random key

        Returns:
            eval_dict: dictionary with evaluation metrics and samples for wandb logging
        """
        eval_dict = {}

        samples = None
        sample_segmentation_mask_approx = None
        inputs = None
        inputs_segmentation_mask = None
        sample_embeddings = {
            embedding: None for embedding in self.auxiliary_data_prep["embedding"]
        }
        mses = []
        path_lengths = []
        nfes = []

        # create vmap functions
        partial_sample_fn = ft.partial(self.sample_fn, model)

        # compute metrics batch-wise
        eval_num_iter = self.num_eval_samples // self.batch_size + 1
        loader = iter(self.dataset)
        
        for _ in tqdm(range(eval_num_iter)):
            src_batch = next(loader)
            # padding for last batch if necessary
            pad_size = self.batch_size - src_batch.data.shape[0]

            inputs_segmentation_mask = jnp_safe_concat(inputs_segmentation_mask, src_batch.segmentation_mask)
            if pad_size > 0:
                src_batch = easy_pad(src_batch, pad_size)

            src_batch = jx_device_put(src_batch, self.shard)

            if self.use_vae:
                inputs = jnp_safe_concat(
                    inputs, self.vae_decode_fn(src_batch.data) * 0.5 + 0.5
                )
            else:
                inputs = jnp_safe_concat(inputs, src_batch.data * 0.5 + 0.5)

            
            ####
            # sample from model
            if self.is_genot:
                batch_size = src_batch.data.shape[0]
                ode_key = jr.split(key, batch_size)
            else:
                ode_key = None

            ###
            # NFE
            sample_batch, nfe = jax.vmap(partial_sample_fn)(src_batch, ode_key)
            nfes.append(nfe)
            ###
            # T - Euclidean distance
            if self.enable_path_lengths:
                # Compute Euclidean distance between samples and inputs
                if pad_size > 0:
                    path_lengths.append(
                        self.rmse_fn(
                            src_batch.data[:-pad_size], sample_batch[:-pad_size]
                        )
                    )
                else:
                    path_lengths.append(self.rmse_fn(src_batch.data, sample_batch))

            ###
            # Decode VAE
            if self.use_vae:
                sample_batch = jx_device_put(sample_batch, self.shard)
                sample_batch = self.vae_decode_fn(sample_batch)
                sample_batch = jnp.clip(sample_batch, -1.0, 1.0)

            if pad_size > 0:
                src_batch = easy_unpad(src_batch, pad_size)
                sample_batch = sample_batch[:-pad_size]

            ####
            # Compute embeddings
            # Make a tolerance > 0.05 as it is more numerically stable to noises 
            sample_segmentation_mask_approx_batch = jnp.expand_dims(
                ((0.5*sample_batch+0.5) > 0.01).any(axis=1), 
                axis=1
            ).astype(src_batch.segmentation_mask.dtype)

            ###
            # Format samples
            samples = jnp_safe_concat(samples, sample_batch)
            sample_segmentation_mask_approx = jnp_safe_concat(sample_segmentation_mask_approx, sample_segmentation_mask_approx_batch)

        for embedding in sample_embeddings:
            embedding_aux = self.auxiliary_data_prep["embedding"][embedding]
            sample_embeddings[embedding] = embedding_aux.embedding_fn(
                    np.asarray(samples),
                    np.asarray(sample_segmentation_mask_approx),
                )
        # Make into [0,1] domain 
        samples = 0.5* samples + 0.5
                
        ###
        # Base metrics
        eval_dict["nfe"] = jnp.mean(jnp.hstack(nfes))
        if self.enable_mse:
            eval_dict["mse"] = jnp.mean(jnp.hstack(mses))
        if self.enable_path_lengths:
            eval_dict["path_lengths"] = jnp.mean(jnp.hstack(path_lengths))
            eval_dict["path_lengths_std"] = jnp.std(jnp.hstack(path_lengths))

        ####
        # Cell metrics 
        eval_dict_cell = self.compute_cell_metrics(
            embeddings=sample_embeddings,
        )
        eval_dict.update(eval_dict_cell)

        ###
        # Sample images
        if self.return_samples:
            wb_image = generate_wb_image(
                samples=samples, inputs=inputs, num_samples=self.num_save_samples
            )
            eval_dict["samples"] = wb_image

            # We also use the approximated one instead of the real one, should be almost identical (takes into account VAE)
            inputs_segmentation_mask_approx =  jnp.expand_dims(
                (inputs > 0.01).any(axis=1), 
                axis=1
            ).astype(src_batch.segmentation_mask.dtype)
            wb_image = generate_wb_image(
                samples=sample_segmentation_mask_approx, inputs=inputs_segmentation_mask_approx, num_samples=self.num_save_samples
            )
            eval_dict["segmentation_mask_samples"] = wb_image


        return eval_dict

    def compute_cell_metrics(
        self,
        embeddings: dict[str, np.ndarray],
    ) -> dict:
        ###
        # Compute cell metrics from embeddings
        eval_dict_cell = dict()
        umap_embeddings = dict()
        for embedding in self.cell_embeddings_metrics:
            ###
            # Extract embeddings
            embedding_value = embeddings[embedding]
            source_embedding_value = self.additional_embeddings["dataset"][embedding]
            target_embedding_value = self.additional_embeddings["target_dataset"][embedding]
            ###

            ###
            # FID
            mu = np.mean(embedding_value, axis=0)
            sigma = np.cov(embedding_value, rowvar=False)
            
            mu_target = np.mean(target_embedding_value, axis=0)
            sigma_target = np.cov(target_embedding_value, rowvar=False)

            mu_source = np.mean(source_embedding_value, axis=0)
            sigma_source = np.cov(source_embedding_value, rowvar=False)

            eval_dict_cell[f"[{embedding}]-FID-target"] = self.compute_fid(
                mu_real=mu_target,
                sigma_real=sigma_target,
                mu_gen=mu,
                sigma_gen=sigma,
            )

            eval_dict_cell[f"[{embedding}]-FID-source"] = self.compute_fid(
                mu_real=mu_source,
                sigma_real=sigma_source,
                mu_gen=mu,
                sigma_gen=sigma,
            )

            ####
            # Precision, Same class Perc & Recall
            eval_dict_cell[f"[{embedding}]-SameClassPerc(K5)-target"] = (
                calculate_same_class_perc(
                    target_embedding_value,
                    embedding_value,
                    top_k=5,
                )
            )

            eval_dict_cell[f"[{embedding}]-Precission(K5)-target"] = estimate_precision(
                target_embedding_value,
                embedding_value,
            )
            eval_dict_cell[f"[{embedding}]-Recall(K5)-target"] = estimate_recall(
                target_embedding_value,
                embedding_value,
            )
        
            ###
            # Generate 2D points for plotting 
            embedding_aux = self.auxiliary_data_prep["embedding"][embedding]
            embedding_2d_projection = embedding_aux["embedding_2d_projection"]

            
            embedding_2d = embedding_2d_projection(embedding_value)
            source_embedding_2d = embedding_2d_projection(source_embedding_value)
            target_embedding_2d = embedding_2d_projection(target_embedding_value)

            umap_embeddings[embedding] = (embedding_2d, source_embedding_2d, target_embedding_2d)

        eval_dict_cell['umap'] = self.make_umap_figure(umap_embeddings)

        for embedding, column_names in self.cell_embeddings_histograms.items():
            # Extract embeddings
            embedding_value = embeddings[embedding]
            source_embedding_value = self.additional_embeddings["dataset"][embedding]
            target_embedding_value = self.additional_embeddings["target_dataset"][embedding]
            ###
            
            embedding_df = np_to_dataframe(embedding_value, columns=column_names)
            source_embedding_df = np_to_dataframe(source_embedding_value, columns=column_names)
            target_embedding_df = np_to_dataframe(target_embedding_value, columns=column_names)
            
            hist_figure = self.make_histograms_figure(dfs={"Source": source_embedding_df, 
                                             "Target": target_embedding_df,
                                             "Generated": embedding_df,
                                            },
                                        colors={"Source": tum_primary, 
                                             "Target": tum_green,
                                             "Generated": tum_red,
                                            })
            eval_dict_cell[f'hist-{embedding}'] = hist_figure

        return eval_dict_cell


    @staticmethod
    def make_histograms_figure(dfs : Dict[str, pd.DataFrame], colors : Optional[Dict[str, pd.DataFrame]] = None):
        colors = colors or dict()
        columns = None 
        for df in dfs.values():
            if columns is None:
                columns = set(df.columns)
            else:
                columns = columns & set(df.columns)
        columns = sorted(columns)
        n_features = len(columns)

        grid_size = math.ceil(math.sqrt(n_features))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(5 * grid_size, 5 * grid_size))
        
        ###
        # Plot each feature
        axes = axes.flatten()
        for i, feature in enumerate(columns):
            ax = axes[i]
            for label, df in dfs.items():
                sns.histplot(df[feature], color=colors.get(label, tum_colors[i % len(tum_colors)]), 
                             kde=False, stat="density", ax=ax, alpha=0.6, label=label)
            ax.set_title(f'{feature}', fontsize=16, fontweight='bold')
            ax.set_ylabel('', fontsize=14)
            ax.set_xlabel('', fontsize=14)
            ax.yaxis.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
            ax.xaxis.grid(False)

        # Hide unused
        for j in range(len(columns), len(axes)):
            fig.delaxes(axes[j])

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=14, title="", title_fontsize=14, frameon=False)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        with io.BytesIO() as buf:
            plt.savefig(buf, format="png")
            buf.seek(0)
            figure = wandb.Image(Image.open(buf))
            plt.close()

        return figure 



    @staticmethod
    def make_umap_figure(plot_embeddings):
        n_embeddings = len(plot_embeddings)
        grid_size = math.ceil(math.sqrt(n_embeddings)) 

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 4, grid_size * 4), dpi=100)
        axes = axes.flatten()  

        ###
        # Plot each embedding 
        for i, (embedding, (embedding_2d, source_embedding_2d, target_embedding_2d)) in enumerate(plot_embeddings.items()):
            ax = axes[i]
            ax.scatter(
                source_embedding_2d[:, 0],
                source_embedding_2d[:, 1],
                s=15,
                color=tum_primary,
                alpha=1,
            )
            ax.scatter(
                target_embedding_2d[:, 0],
                target_embedding_2d[:, 1],
                s=15,
                color=tum_green,
                alpha=0.3,
            )
            ax.scatter(
                embedding_2d[:, 0],
                embedding_2d[:, 1],
                s=20,
                color=tum_red,
                marker='x'
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_box_aspect(1) 
            ax.set_title(f"{embedding}", fontsize=12, fontweight='bold')

        # Remove bounding boxed 
        for j in range(len(axes)): # range(i + 1, len(axes)) <- Only remove from not umap plots 
            axes[j].axis('off')


        fig.legend(
            labels=["Source", "Target", "Generated"],
            loc="upper center",
            # bbox_to_anchor=(0.5, 1.00),
            ncol=3,
            frameon=False,
            fontsize=16
        )

        fig.tight_layout(rect=[0, 0, 1, 0.9])  
        with io.BytesIO() as buf:
            plt.savefig(buf, format="png")
            buf.seek(0)
            figure = wandb.Image(Image.open(buf))
            plt.close()
        return figure 

    @staticmethod
    def compute_fid(
        mu_real: np.ndarray,
        sigma_real: np.ndarray,
        mu_gen: np.ndarray,
        sigma_gen: np.ndarray,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """
        Compute Frechet Inception Distance (FID) between two distributions.
        """
        # compute statistics
        mu_gen = np.atleast_1d(mu_gen)
        mu_real = np.atleast_1d(mu_real)
        sigma_gen = np.atleast_1d(sigma_gen)
        sigma_real = np.atleast_1d(sigma_real)

        assert (
            mu_gen.shape == mu_real.shape
        ), f"Shapes {mu_gen.shape} != {mu_real.shape}"
        assert (
            sigma_gen.shape == sigma_real.shape
        ), f"Shapes {sigma_gen.shape} != {sigma_real.shape}"

        diff = mu_real - mu_gen
        covmean, _ = scipy.linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)

        if not np.isfinite(covmean).all():
            warnings.warn(
                (
                    f"fid calculation produces singular product; "
                    "adding {eps} to diagonal of cov estimates"
                )
            )
            offset = np.eye(sigma_real.shape[0]) * eps
            covmean = scipy.linalg.sqrtm((sigma_real + offset).dot(sigma_gen + offset))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-2):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component {m}")
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        return (
            diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_gen) - 2 * tr_covmean
        )
