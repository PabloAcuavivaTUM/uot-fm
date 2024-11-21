import csv
import glob
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from ml_collections import ConfigDict
from tqdm import tqdm

from models import get_clip_fns
from utils import GenerationSampler

from .miscellaneous import EasyDict
from .transforms import low_pass_filter 

# TODO: Once verified the code is properly working, change "map_forward" so that is works for all datasets in translation, simply switch
#  the source and the target dataset after getting the data if need be and remove all references thereafter
# TODO: Overfit to one batch can be mostly substituted in many places for nsamples (Leave it as it changes some preprocessing, but the function
# which gets the data does not need it).


def get_translation_datasets(
    config: ConfigDict,
    shard: Optional[jax.sharding.Sharding] = None,
    vae_encode_fn: Optional[Callable] = None,
) -> List[tf.data.Dataset]:
    """Get translation datasets and prepare them."""
    train_source, train_target, eval_source, eval_target, auxiliary_data_prep = (
        get_data(config, shard, vae_encode_fn)
    )
    if config.data.low_pass_filter:
        train_source["low_freq_data"] = tf.stack([low_pass_filter(d, **config.data.low_pass_filter) for d in train_source.data])
        train_target["low_freq_data"] = tf.stack([low_pass_filter(d, **config.data.low_pass_filter) for d in train_target.data])
        eval_source["low_freq_data"] = tf.stack([low_pass_filter(d, **config.data.low_pass_filter) for d in eval_source.data])
        eval_target["low_freq_data"] = tf.stack([low_pass_filter(d, **config.data.low_pass_filter) for d in eval_target.data])

    train_source_ds = prepare_dataset(train_source, config)
    eval_source_ds = prepare_dataset(eval_source, config, evaluation=True)
    train_target_ds = prepare_dataset(train_target, config)
    eval_target_ds = prepare_dataset(eval_target, config, evaluation=True)
    return (
        train_source_ds,
        train_target_ds,
        eval_source_ds,
        eval_target_ds,
        auxiliary_data_prep,
    )


def prepare_dataset(
    data: EasyDict,
    config: ConfigDict,
    evaluation: bool = False,
) -> tfds.as_numpy:
    """Prepare dataset given config."""
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.map(
        get_preprocess_fn(config, evaluation),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    if not evaluation:
        dataset = dataset.shuffle(config.data.shuffle_buffer)
        dataset = dataset.repeat()

    # Notice the distinction between evaluation and not evaluation
    batch_size = (
        config.training.batch_size_matching
        if not evaluation
        else config.training.batch_size
    )
    dataset = dataset.batch(batch_size, drop_remainder=not evaluation)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = tfds.as_numpy(dataset)
    dataset.length = data.data.shape[0]
    return dataset


def get_preprocess_fn(config, evaluation: bool = False, precomputing: bool = False):
    """Get preprocessing function for dataset."""

    def process_ds(x: np.ndarray) -> tf.Tensor:
        ###
        # Quick hack for campa images 
        if config.data.source == "campa_cell":
            return tf.cast(x, tf.float32)

        ###
        # Normal image process func
        x = tf.cast(x, tf.float32) / 127.5 - 1.0
        if config.data.source == "celeba_attribute":
            x = tf.image.resize(x, config.data.shape[1:], antialias=True)
            if config.data.random_crop:
                if not evaluation and not config.overfit_to_one_batch:
                    x = tf.image.random_crop(x, size=config.data.crop_shape)
                else:
                    x = central_crop(x, size=config.data.crop_shape[0])
            x = tf.transpose(x, perm=[2, 0, 1])
        elif config.data.source in "horse2zebra":
            x = tf.image.resize(x, config.data.shape[1:], antialias=True)
            x = tf.transpose(x, perm=[2, 0, 1])
        elif config.task == "generation":
            x = tf.image.random_flip_left_right(x)
            x = tf.transpose(x, perm=[2, 0, 1])

        return x

    if not precomputing:
        if config.model.use_vae:
            process_ds = lambda x: tf.cast(x, tf.float32)
        return lambda easydict: EasyDict(
            data=process_ds(easydict.pop("data")), **easydict
        )

    return process_ds


def central_crop(image: tf.Tensor, size: int) -> tf.Tensor:
    """Crop the center of an image to the given size."""
    top = (image.shape[0] - size) // 2
    left = (image.shape[1] - size) // 2
    return tf.image.crop_to_bounding_box(image, top, left, size, size)


def get_data(
    config: ConfigDict,
    shard: Optional[jax.sharding.Sharding] = None,
    vae_encode_fn: Optional[Callable] = None,
) -> List[Union[np.ndarray, Dict[str, np.ndarray]]]:
    """Load source and target, train and evaluation data."""
    auxiliary_data_prep = None

    if vae_encode_fn is not None:
        preprocess_fn = get_preprocess_fn(config, evaluation=True, precomputing=True)
    else:
        preprocess_fn = None

    if config.data.target == "emnist":
        train_source, train_target = emnist("train")
        eval_source, eval_target = emnist("test")
    elif config.data.target == "celeba_attribute":
        train_source, train_target = celeba_attribute(
            "train",
            config.data.attribute_id,
            config.data.map_forward,
            config.training.batch_size,
            config.overfit_to_one_batch,
            shard,
            vae_encode_fn,
            preprocess_fn,
            additional_embedding=config.data.additional_embedding,
            nsamples=config.data.get("nsamples", None),
        )
        eval_source, eval_target = celeba_attribute(
            "test",
            config.data.attribute_id,
            config.data.map_forward,
            config.training.batch_size,
            config.overfit_to_one_batch,
            shard,
            vae_encode_fn,
            preprocess_fn,
            additional_embedding=config.data.additional_embedding,
            nsamples=config.data.get("nsamples", None),
        )
    elif config.data.target == "horse2zebra":
        train_source, train_target = horse2zebra(
            split="train",
            batch_size=config.training.batch_size,
            overfit_to_one_batch=config.overfit_to_one_batch,
            shard=shard,
            vae_encode_fn=vae_encode_fn,
            preprocess_fn=preprocess_fn,
            additional_embedding=config.data.additional_embedding,
        )

        eval_source, eval_target = horse2zebra(
            split="test",
            batch_size=config.training.batch_size,
            overfit_to_one_batch=config.overfit_to_one_batch,
            shard=shard,
            vae_encode_fn=vae_encode_fn,
            preprocess_fn=preprocess_fn,
            additional_embedding=config.data.additional_embedding,
        )
    elif config.data.target == "gaussian":
        # TODO: Not checked if it works
        train_source, train_target = get_unbalanced_uniform_samplers(
            input_dim=config.input_dim,
            num_samples=config.num_samples,
        )
        eval_source, eval_target = get_unbalanced_uniform_samplers(
            input_dim=config.input_dim,
            num_samples=config.eval.eval_samples,
        )

    elif config.data.target == "celeba_fake":
        # Fake datata with same dimensions as celeba256 encoded for quick pipeline prototyping
        train_source, train_target = celeba_fake(
            "train",
            config.data.attribute_id,
            config.data.map_forward,
            config.training.batch_size,
            additional_embedding=config.data.additional_embedding,
        )
        eval_source, eval_target = celeba_fake(
            "test",
            config.data.attribute_id,
            config.data.map_forward,
            config.training.batch_size,
            additional_embedding=config.data.additional_embedding,
        )
    elif config.data.target == "campa_cell":
        train_source, train_target, eval_source, eval_target, auxiliary_data_prep = (
            campa_cell(
                type_src=config.data.type_src,
                type_tgt=config.data.type_tgt,
                batch_size=config.training.batch_size,
                shard=shard,
                vae_encode_fn=vae_encode_fn,
                preprocess_fn=preprocess_fn,
                channels=config.data.channels,
                additional_embedding=config.data.additional_embedding,
                embedding_combinations=config.data.embedding_combinations,
            )
        )
    else:
        raise ValueError(f"Unknown target dataset {config.target.data}")

    # for translation between different datasets. Not implemented. For now target = source
    if config.data.source == "gaussian":
        pass
    elif config.data.source == "celeba_attribute":
        pass
    elif config.data.source == "horse2zebra":
        pass
    elif config.data.source == "emnist":
        pass
    elif config.data.source == "celeba_fake":
        pass
    elif config.data.source == "campa_cell":
        pass
    else:
        raise ValueError(f"Unknown source dataset {config.data.source}")

    if config.overfit_to_one_batch:
        train_source = train_source.slice(slice(0, config.training.batch_size))
        train_target = train_target.slice(slice(0, config.training.batch_size))
        eval_source = train_source.slice(slice(0, config.training.batch_size))
        eval_target = train_target.slice(slice(0, config.training.batch_size))

    return (
        train_source,
        train_target,
        eval_source,
        eval_target,
        auxiliary_data_prep,
    )


def emnist(split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load emnist data from numpy files."""
    target_dir = os.getcwd() + "/data/emnist"
    if split == "train":
        data_x = np.load(f"{target_dir}/x_train.npy")
        data_y = np.load(f"{target_dir}/y_train.npy")
    elif split == "test":
        data_x = np.load(f"{target_dir}/x_test.npy")
        data_y = np.load(f"{target_dir}/y_test.npy")
    elif split == "full":
        data_x = np.concatenate(
            [np.load(f"{target_dir}/x_train.npy"), np.load(f"{target_dir}/x_test.npy")]
        )
        data_y = np.concatenate(
            [np.load(f"{target_dir}/y_train.npy"), np.load(f"{target_dir}/y_test.npy")]
        )
    digits_indices = np.isin(data_y, np.array([0, 1, 8]))
    letters_indices = np.logical_not(digits_indices)
    source_data = data_x[digits_indices]
    target_data = data_x[letters_indices]
    # Map labels to 0, 1, 2
    map_fn = np.vectorize({0: 0, 1: 1, 8: 2, 11: 2, 18: 1, 24: 0}.__getitem__)
    data_y = map_fn(data_y)
    source_label = data_y[digits_indices]
    target_label = data_y[letters_indices]
    one_hot_src_labels = np.eye(3)[source_label]
    one_hot_tgt_labels = np.eye(3)[target_label]

    source = EasyDict(data=source_data, label=one_hot_src_labels)
    target = EasyDict(data=target_data, label=one_hot_tgt_labels)

    return source, target


def celeba_fake(
    split: str,
    attribute_id: int,
    map_forward: bool,
    batch_size: int,
    subset_attribute_id: Optional[int] = None,
    additional_embedding: Optional[str] = None,
):
    data_dir = "./data/celeba"
    with open(f"{data_dir}/list_attr_celeba.txt") as csv_file:
        data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))
        data = data[2:]
        filenames = [row[0] for row in data]
        data = [row[1:] for row in data]
        label_int = np.array([list(map(int, i)) for i in data])

    with open(f"{data_dir}/list_eval_partition.txt") as csv_file:
        data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))
        data = [row[1:] for row in data]
        split_int = np.array([list(map(int, i)) for i in data])
    if split == "train":
        splits = 0
    elif split == "test":
        splits = [1, 2]
    elif split == "full":
        splits = [0, 1, 2]
    split_indices = np.isin(split_int, splits).squeeze()
    if map_forward:
        source_indices = label_int[:, attribute_id] != 1
        target_indices = label_int[:, attribute_id] == 1
    else:
        source_indices = label_int[:, attribute_id] == 1
        target_indices = label_int[:, attribute_id] != 1
    if subset_attribute_id is not None:
        if subset_attribute_id == 201:
            # subset for glasses
            source_indices = source_indices * (label_int[:, 20] != 1)
            target_indices = target_indices * (label_int[:, 20] != 1)
        else:
            source_indices = source_indices * (label_int[:, subset_attribute_id] == 1)
            target_indices = target_indices * (label_int[:, subset_attribute_id] == 1)

    source_indices = split_indices * source_indices
    target_indices = split_indices * target_indices
    source_labels = np.array(
        [label for label, indice in zip(label_int, source_indices) if indice]
    )
    target_labels = np.array(
        [label for label, indice in zip(label_int, target_indices) if indice]
    )

    N = min(512 * 16, source_labels.shape[0], target_labels.shape[0])
    target_data = jnp.abs(np.random.rand(N, 4, 32, 32))
    source_data = jnp.abs(np.random.rand(N, 4, 32, 32))

    source = EasyDict(data=source_data, label=source_labels[:N])
    target = EasyDict(data=target_data, label=target_labels[:N])

    if additional_embedding:
        source["embedding"] = jnp.abs(np.random.rand(N, 512))
        target["embedding"] = jnp.abs(np.random.rand(N, 512))

    return source, target


def celeba_attribute(
    split: str,
    attribute_id: int,
    map_forward: bool,
    batch_size: int,
    overfit_to_one_batch: bool,
    shard: Optional[jax.sharding.Sharding] = None,
    vae_encode_fn: Optional[Callable] = None,
    preprocess_fn: Optional[Callable] = None,
    subset_attribute_id: Optional[int] = None,
    additional_embedding: Optional[str] = None,
    nsamples: Optional[int] = None,
) -> Tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Load celeba attribute data.

    Args:
        split: Train, test or full split
        attribute_id: Attribute id to split on (0-39)
        map_forward: Whether to map forward or backward
        batch_size: Batch size
        overfit_to_one_batch: Whether to overfit to one batch
        shard: Sharding object for vae encoding
        vae_encode_fn: Vae encode function
        preprocess_fn: Preprocess function
        subset_attribute_id: Subset attribute id to split on (0-39)
        nsamples: Indicates the number of samples to load. Default None, load all.
    """
    data_dir = "./data/celeba"
    with open(f"{data_dir}/list_attr_celeba.txt") as csv_file:
        data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))
        data = data[2:]
        filenames = [row[0] for row in data]
        data = [row[1:] for row in data]
        label_int = np.array([list(map(int, i)) for i in data])

    with open(f"{data_dir}/list_eval_partition.txt") as csv_file:
        data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))
        data = [row[1:] for row in data]
        split_int = np.array([list(map(int, i)) for i in data])

    # get indices for split and attribute
    if split == "train":
        splits = 0
    elif split == "test":
        splits = [1, 2]
    elif split == "full":
        splits = [0, 1, 2]
    split_indices = np.isin(split_int, splits).squeeze()
    if map_forward:
        source_indices = label_int[:, attribute_id] != 1
        target_indices = label_int[:, attribute_id] == 1
    else:
        source_indices = label_int[:, attribute_id] == 1
        target_indices = label_int[:, attribute_id] != 1
    if subset_attribute_id is not None:
        if subset_attribute_id == 201:
            # subset for glasses
            source_indices = source_indices * (label_int[:, 20] != 1)
            target_indices = target_indices * (label_int[:, 20] != 1)
        else:
            source_indices = source_indices * (label_int[:, subset_attribute_id] == 1)
            target_indices = target_indices * (label_int[:, subset_attribute_id] == 1)

    # get filenames
    source_indices = split_indices * source_indices
    target_indices = split_indices * target_indices
    source_filenames = [
        filename for filename, indice in zip(filenames, source_indices) if indice
    ]
    source_labels = np.array(
        [label for label, indice in zip(label_int, source_indices) if indice]
    )
    target_filenames = [
        filename for filename, indice in zip(filenames, target_indices) if indice
    ]
    target_labels = np.array(
        [label for label, indice in zip(label_int, target_indices) if indice]
    )

    if nsamples is not None:
        source_filenames = source_filenames[:nsamples]
        source_labels = source_labels[:nsamples]
        target_filenames = target_filenames[:nsamples]
        target_labels = target_labels[:nsamples]

    logging.info("Loading source and target data.")
    source_data = []
    target_data = []

    # Load source data
    for fname in tqdm(source_filenames):
        image = cv2.imread(f"{data_dir}/img_align_celeba/{fname}")
        # cv2 reads images in BGR format, so we need to reverse the channel
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        source_data.append(image)
        if overfit_to_one_batch and len(source_data) == batch_size:
            break

    # Load target data
    for fname in tqdm(target_filenames):
        image = cv2.imread(f"{data_dir}/img_align_celeba/{fname}")
        # cv2 reads images in BGR format, so we need to reverse the channel
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target_data.append(image)
        if overfit_to_one_batch and len(target_data) == batch_size:
            break

    if additional_embedding:
        source_embedding = compute_embedding(
            source_data, embedding=additional_embedding
        )
        target_embedding = compute_embedding(
            target_data, embedding=additional_embedding
        )

    if vae_encode_fn is not None:
        logging.info("Preprocessing for VAE embedding.")
        source_data = [preprocess_fn(image).numpy() for image in source_data]
        target_data = [preprocess_fn(image).numpy() for image in target_data]
        logging.info("Precomputing VAE embedding.")
        source_data = compute_vae_encoding(
            source_data, vae_encode_fn=vae_encode_fn, batch_size=batch_size, shard=shard
        )
        target_data = compute_vae_encoding(
            target_data, vae_encode_fn=vae_encode_fn, batch_size=batch_size, shard=shard
        )
    else:
        source_data = np.array(source_data)
        target_data = np.array(target_data)

    source = EasyDict(data=source_data, label=source_labels)
    target = EasyDict(data=target_data, label=target_labels)

    if additional_embedding:
        source["embedding"] = source_embedding
        target["embedding"] = target_embedding

    return source, target
    # return source_data, target_data, source_labels, target_labels


def horse2zebra(
    split: str,
    batch_size: int,
    overfit_to_one_batch: bool,
    shard: Optional[jax.sharding.Sharding] = None,
    vae_encode_fn: Optional[Callable] = None,
    preprocess_fn: Optional[Callable] = None,
    additional_embedding: Optional[str] = None,
    nsamples: Optional[int] = None,
) -> Tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Load horse2zebra data.

    Args:
        split: Train, test or full split
        batch_size: Batch size
        overfit_to_one_batch: Whether to overfit to one batch
        shard: Sharding object for vae encoding
        vae_encode_fn: Vae encode function
        preprocess_fn: Preprocess function
        subset_attribute_id: Subset attribute id to split on (0-39)
        nsamples: Indicates the number of samples to load. Default None, load all.
    """

    data_dir = "./data/horse2zebra"
    if split == "train":
        source_filenames = glob.glob(os.path.join(data_dir, "trainA", "*"))
        target_filenames = glob.glob(os.path.join(data_dir, "trainB", "*"))
    elif split == "test":
        source_filenames = glob.glob(os.path.join(data_dir, "testA", "*"))
        target_filenames = glob.glob(os.path.join(data_dir, "testB", "*"))
    elif split == "full":
        source_filenames = glob.glob(os.path.join(data_dir, "trainA", "*"))
        target_filenames = glob.glob(os.path.join(data_dir, "trainB", "*"))

        source_filenames += glob.glob(os.path.join(data_dir, "testA", "*"))
        target_filenames += glob.glob(os.path.join(data_dir, "testB", "*"))

    if nsamples is not None:
        source_filenames = source_filenames[:nsamples]
        target_filenames = target_filenames[:nsamples]

    logging.info("Loading source and target data.")
    source_data = []
    target_data = []

    # Load source data
    for fname in tqdm(source_filenames):
        image = cv2.imread(fname)
        # cv2 reads images in BGR format, so we need to reverse the channel
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        source_data.append(image)
        if overfit_to_one_batch and len(source_data) == batch_size:
            break

    # Load target data
    for fname in tqdm(target_filenames):
        image = cv2.imread(fname)
        # cv2 reads images in BGR format, so we need to reverse the channel
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target_data.append(image)
        if overfit_to_one_batch and len(target_data) == batch_size:
            break

    if additional_embedding:
        source_embedding = compute_embedding(
            source_data, embedding=additional_embedding
        )
        target_embedding = compute_embedding(
            target_data, embedding=additional_embedding
        )

    if vae_encode_fn is not None:
        logging.info("Preprocessing for VAE embedding.")
        source_data = [preprocess_fn(image).numpy() for image in source_data]
        target_data = [preprocess_fn(image).numpy() for image in target_data]
        logging.info("Precomputing VAE embedding.")
        source_data_vae = compute_vae_encoding(
            source_data, vae_encode_fn=vae_encode_fn, batch_size=batch_size, shard=shard
        )
        target_data_vae = compute_vae_encoding(
            target_data, vae_encode_fn=vae_encode_fn, batch_size=batch_size, shard=shard
        )

        source = EasyDict(
            data=source_data_vae
        )  # , original_data=np.array(source_data))
        target = EasyDict(
            data=target_data_vae
        )  # , original_data=np.array(target_data))

    else:
        source_data = np.array(source_data)
        target_data = np.array(target_data)

        source = EasyDict(data=source_data)
        target = EasyDict(data=target_data)

    if additional_embedding:
        source["embedding"] = source_embedding
        target["embedding"] = target_embedding

    return source, target


def compute_vae_encoding(
    data: list[np.ndarray],
    vae_encode_fn: Callable,
    batch_size: int,
    shard: Optional[jax.sharding.Sharding] = None,
) -> np.ndarray:

    batch_size = batch_size // 2
    vae_data = []
    # compute vae embedding batch-wise
    for idx in tqdm(range(0, len(data), batch_size)):
        batch = np.array(data[idx : idx + batch_size])
        num_pad = batch_size - batch.shape[0]
        if batch.shape[0] < batch_size:
            # pad batch, shard and then unpad again
            batch = np.concatenate([batch, np.zeros([num_pad, *batch.shape[1:]])])
        batch = jax.device_put(batch, shard)
        vae_out = vae_encode_fn(batch)
        if num_pad > 0:
            vae_out = vae_out[:-num_pad]
        vae_data.append(vae_out)

    return np.concatenate(vae_data)


def compute_embedding(data: list[np.ndarray], embedding: str):
    if embedding == "clip":
        encode_img_fn, encode_text_fn = get_clip_fns()
        embedded_data = encode_img_fn(data)
    else:
        raise ValueError(f"embedding {embedding} is not valid.")
    return embedded_data


def get_unbalanced_uniform_samplers(
    input_dim: int = 2,
    num_samples: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate unbalanced Gaussian data and return a tuple of data samplers."""
    # generate source data
    source_center_one = np.repeat(
        np.array([0, -1])[None, :], int(num_samples * 1.5), axis=0
    )
    source_center_two = np.repeat(np.array([5, -1])[None, :], num_samples, axis=0)
    source_center = np.concatenate([source_center_one, source_center_two])
    source_data = source_center + np.random.uniform(
        size=[int(num_samples * 1.5) + num_samples, input_dim], low=-0.5, high=0.5
    )
    # generate target data
    target_center_one = np.repeat(np.array([0, 1])[None, :], num_samples, axis=0)
    target_center_two = np.repeat(
        np.array([5, 1])[None, :], int(num_samples * 1.5), axis=0
    )
    target_center = np.concatenate([target_center_one, target_center_two])
    target_data = target_center + np.random.uniform(
        size=[int(num_samples * 1.5) + num_samples, input_dim], low=-0.5, high=0.5
    )

    source = EasyDict(data=source_data)
    target = EasyDict(data=target_data)

    return source, target


# ------------------
# TODO: It does not work as it won't return an easydict, unify generation and translation. For now deactive generation
def get_generation_datasets(config: ConfigDict) -> GenerationSampler:
    """Get generation dataset and create sampler."""
    train_data = cifar10("train")
    return GenerationSampler(jnp.array(train_data), config.training.batch_size_matching)


def cifar10(split: str) -> np.ndarray:
    """Load cifar10 data from tensorflow datasets."""
    [x_train, y_train], [x_test, y_test] = tf.keras.datasets.cifar10.load_data()
    if split == "train":
        return x_train
    else:
        return x_test


####
# Cell datasets
import glob
from copy import deepcopy
import umap

from utils.cell_fns.features import (
    calculate_intensity_features,
    calculate_morphological_features,
)


def compute_cell_embeddings(
    obj_imgs: np.ndarray,
    segmentation_masks: np.ndarray,
    embedding_type: str,
    embedding_kwargs,
    idchannel : Optional[int] = None
):
    # Most likely unnecesary, but we make sure they stay frozen
    embedding_kwargs = deepcopy(
        embedding_kwargs
    )  

    def format_input( obj_imgs : np.ndarray, segmentation_masks: np.ndarray):
        obj_imgs = obj_imgs.transpose(0,2,3,1)
        segmentation_masks = segmentation_masks.transpose(0,2,3,1)
        if idchannel is not None:
            obj_imgs = obj_imgs[:,:,:,idchannel:idchannel+1]
        return obj_imgs, segmentation_masks
    
    obj_imgs, segmentation_masks = format_input(obj_imgs, segmentation_masks)

    if embedding_type == "morphological_features":
        embedding_value = calculate_morphological_features(
            segmentation_masks, features_list=embedding_kwargs["features_list"]
        )
        embedding_value = np.array([emb.to_array() for emb in embedding_value])

        def embedding_fn(obj_imgs, segmentation_masks): 
            embedding_value = calculate_morphological_features(
                segmentation_masks, features_list=embedding_kwargs["features_list"]
            )
            embedding_value = np.array([emb.to_array() for emb in embedding_value])
            return embedding_value


        
    elif embedding_type == "channel_features":
        embedding_value = calculate_intensity_features(
            obj_imgs, features_list=embedding_kwargs["features_list"]
        )
        embedding_value = np.array([emb.to_array() for emb in embedding_value])

        def embedding_fn(obj_imgs, segmentation_masks): 
            embedding_value = calculate_intensity_features(
                obj_imgs, features_list=embedding_kwargs["features_list"]
            )
            embedding_value = np.array([emb.to_array() for emb in embedding_value])
            return embedding_value



    elif embedding_type == "morphological_umap":
        umap_model = umap.UMAP(**embedding_kwargs)
        embedding_value = umap_model.fit_transform(
            segmentation_masks.reshape(segmentation_masks.shape[0], -1)
        )
        embedding_fn = lambda obj_imgs, segmentation_masks: umap_model.transform(
            segmentation_masks.reshape(segmentation_masks.shape[0], -1)
        )
    elif embedding_type == "channel_umap":
        umap_model_channels = []
        embedding_values = []
        n_channels = obj_imgs.shape[-1]
        for ichannel in range(n_channels):
            iobj_imgs = obj_imgs[:, :, :, ichannel]
            umap_model = umap.UMAP(**embedding_kwargs)
            _embedding_value = umap_model.fit_transform(
                iobj_imgs.reshape(iobj_imgs.shape[0], -1)
            )

            embedding_values.append(_embedding_value)
            umap_model_channels.append(umap_model)

        embedding_value = np.concatenate(embedding_values, axis=1)
    
        def embedding_fn(obj_imgs, segmentation_masks):
            embedding_values = []
            for ichannel in range(n_channels):
                umap_model = umap_model_channels[ichannel]
                iobj_imgs = obj_imgs[:, :, :, ichannel]
                embedding_value = umap_model.transform(
                    iobj_imgs.reshape(iobj_imgs.shape[0], -1)
                )

                embedding_values.append(embedding_value)
            return np.concatenate(embedding_values, axis=1)
    else: 
        raise ValueError(f'Invalid cell embedding {embedding_type} asked for.')

    
    # Corrected embedding function 
    def _embedding_fn(obj_imgs : np.ndarray, segmentation_masks : np.ndarray):
        obj_imgs, segmentation_masks = format_input(obj_imgs, segmentation_masks)        
        return embedding_fn(obj_imgs, segmentation_masks) 

    # Generate embedding 2D projection for eval plotting
    if embedding_value.shape[1] == 2:
        embedding_2d_projection = lambda emb: emb
    else:
        umap_2d = umap.UMAP(n_components=2, random_state=42)
        _ = umap_2d.fit_transform(embedding_value)
        embedding_2d_projection = lambda emb: umap_2d.transform(emb)

    return embedding_value, EasyDict(
        embedding_fn=_embedding_fn,
        embedding_2d_projection=embedding_2d_projection,
    )


def campa_cell(
    type_src: str,
    type_tgt: str,
    batch_size: int,
    shard: Optional[jax.sharding.Sharding] = None,
    vae_encode_fn: Optional[Callable] = None,
    preprocess_fn: Optional[Callable] = None,
    channels: Optional[str] = None,
    additional_embedding: Optional[Dict[str, Dict[str, Any]]] = None,
    embedding_combinations : Optional[Dict[str,list[str]]] = None,
    n_src_perc : float = 0.75,
    n_tgt_perc : float = 0.66, 
) -> Tuple[
    EasyDict,
    EasyDict,
    EasyDict,
]:
    data_dir = "/lustre/groups/ml01/workspace/fm_cv/np_campa"
    additional_embedding = additional_embedding or dict()
    embedding_combinations = embedding_combinations or dict()

    # Prepare src and target for dataset
    dataset = dict()
    auxiliary_data_prep = EasyDict()
    for i, type_name in [("src", type_src), ("tgt", type_tgt)]:
        # Load type of data for each well and assign to i (either src or tgt)
        obj_imgs_wells = []
        segmentation_masks_wells = []

        for well_path in glob.glob(os.path.join(data_dir, type_name, "*")):
            obj_imgs = None 
            for channel in channels:
                objs_channel = np.load(os.path.join(well_path, f"{channel}.npy"))
                if obj_imgs is None:
                    obj_imgs = objs_channel
                else:
                    obj_imgs = np.concatenate((obj_imgs, objs_channel), axis=-1) 
            obj_imgs_wells.append(obj_imgs)
            segmentation_masks = np.load(os.path.join(well_path, f"segmentation_masks.npy"))
            segmentation_masks_wells.append(segmentation_masks)
        obj_imgs_wells = np.concatenate(obj_imgs_wells)
        segmentation_masks_wells = np.concatenate(segmentation_masks_wells)

        C = len(channels)
        obj_imgs_wells_max = np.max(obj_imgs_wells.reshape(-1, C), axis=0).reshape(
            1, 1, 1, C
        )

        # Make images into [-1,1] range
        obj_imgs_wells = 2.0 * (
            obj_imgs_wells / obj_imgs_wells_max - 0.5
        )  

        ###
        # [FILTERING] Remove outliers and errors, cell too small to be seen
        npixels = np.prod(segmentation_masks_wells[0].shape)
        too_small_mask = np.array([segmentation_mask.sum() / npixels < 0.05 for segmentation_mask in segmentation_masks_wells])

        obj_imgs_wells = obj_imgs_wells[~too_small_mask]
        segmentation_masks_wells = segmentation_masks_wells[~too_small_mask]
        ###

        dataset[i] = (obj_imgs_wells, segmentation_masks_wells)
        auxiliary_data_prep[i] = dict(max=obj_imgs_wells_max)

    N_src = len(dataset["src"][0])
    N_tgt = len(dataset["tgt"][0])

    obj_imgs_both = np.concatenate([dataset["src"][0], dataset["tgt"][0]])
    segmentation_masks_both = np.concatenate([dataset["src"][1], dataset["tgt"][1]])
    segmentation_masks_both = segmentation_masks_both.transpose(0,3,1,2) # [B, C, H, W]
    obj_imgs_both = obj_imgs_both.transpose(0,3,1,2) # [B, C, H, W]
    # TODO: Any data augmentation: Rotations?
    

    embeddings = dict()
    aux_embedding =  dict()
    for embedding_name, embedding_kwargs in additional_embedding.items():
        # Hack to only give one specific channel embedding
        if '__' in embedding_name:
            embedding_type, embedding_channels = embedding_name.split('__')
        else:
            embedding_type, embedding_channels = embedding_name, ''

        idchannel = None
        if embedding_channels:
            embedding_channels = set(embedding_channels.split('|'))
            # TODO: If we want to add more than one per embedding, don't use pop, also fix above in compute_cell_embedings to get proper embeddings
            idchannel = [i for i, item in enumerate(channels) if item in embedding_channels].pop()
            
            
        embedding_value, embedding_aux = compute_cell_embeddings(
            obj_imgs_both,
            segmentation_masks_both,
            embedding_type=embedding_type,
            embedding_kwargs=embedding_kwargs,
            idchannel=idchannel,
        )
        embeddings[embedding_name] = embedding_value
        aux_embedding[embedding_name] = embedding_aux
            
    auxiliary_data_prep['embedding'] = aux_embedding
    
    for embedding_combination_name, embedding_names in embedding_combinations.items():
        embedding_combination = None 
        for embedding_name in embedding_names:
            if embedding_combination is None:
                embedding_combination = embeddings[embedding_name]
            else:        
                embedding_combination = np.concatenate((embedding_combination, embeddings[embedding_name]), axis=-1)
        embeddings[embedding_combination_name] = embedding_combination

    if vae_encode_fn is not None:
        preprocessed_obj_imgs_both = [
            preprocess_fn(obj_img) for obj_img in obj_imgs_both
        ]
        obj_imgs_both_vae = compute_vae_encoding(
            preprocessed_obj_imgs_both,
            vae_encode_fn=vae_encode_fn,
            batch_size=batch_size,
            shard=shard,
        )
        src_data = EasyDict(data=obj_imgs_both_vae[:N_src]) # , uncompressed_data=obj_imgs_both[:N_src])
        tgt_data = EasyDict(data=obj_imgs_both_vae[N_src:]) # , uncompressed_data=obj_imgs_both[N_src:])
    else:
        src_data = EasyDict(data=obj_imgs_both[:N_src])
        tgt_data = EasyDict(data=obj_imgs_both[N_src:])

    src_data["segmentation_mask"] = segmentation_masks_both[:N_src]
    tgt_data["segmentation_mask"] = segmentation_masks_both[N_src:]

    for embedding, embedding_value in embeddings.items():
        src_data[embedding] = embedding_value[:N_src]
        tgt_data[embedding] = embedding_value[N_src:]
    
    n_src_train = int(n_src_perc * N_src)

    train_src = EasyDict(**{k: v[:n_src_train] for k, v in src_data.items()})
    eval_src = EasyDict(**{k: v[n_src_train:] for k, v in src_data.items()})

    # Eval target is not used, therefore to get a bit more target data, we just copy the train tgt dataset for it
    n_tgt_train = int(n_tgt_perc * N_tgt)
    train_tgt = EasyDict(**{k: v[:n_tgt_train] for k, v in tgt_data.items()})
    eval_tgt = EasyDict(**{k: v[n_tgt_train:] for k, v in tgt_data.items()})

    return train_src, train_tgt, eval_src, eval_tgt, auxiliary_data_prep
