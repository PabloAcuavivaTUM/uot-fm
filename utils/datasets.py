import csv
import logging
import os
import glob 
from typing import Callable, List, Dict, Optional, Tuple, Union 

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from ml_collections import ConfigDict
from tqdm import tqdm

from utils import GenerationSampler
from .miscellaneous import EasyDict
from models import get_clip_fns


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
    train_source, train_target, eval_source, eval_target = get_data(
        config, shard, vae_encode_fn
    )
    train_source_ds = prepare_dataset(train_source, config)
    eval_source_ds = prepare_dataset(eval_source, config, evaluation=True)
    train_target_ds = prepare_dataset(train_target, config)
    eval_target_ds = prepare_dataset(eval_target, config, evaluation=True)
    return train_source_ds, train_target_ds, eval_source_ds, eval_target_ds


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
    dataset = dataset.batch(config.training.batch_size, drop_remainder=not evaluation)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = tfds.as_numpy(dataset)
    dataset.length = data.data.shape[0]
    return dataset


def get_preprocess_fn(config, evaluation: bool = False, precomputing: bool = False):
    """Get preprocessing function for dataset."""

    def process_ds(x: np.ndarray) -> tf.Tensor:
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
        return lambda easydict: EasyDict(data=process_ds(easydict.pop('data')), **easydict)
    
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
    
    if vae_encode_fn is not None:
            preprocess_fn = get_preprocess_fn(
                config, evaluation=True, precomputing=True
            )
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
            additional_embedding=config.data.additional_embedding
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
            additional_embedding=config.data.additional_embedding
        )
    elif config.data.target == "horse2zebra":
        train_source, train_target = horse2zebra(
            split="train",
            batch_size=config.training.batch_size,
            overfit_to_one_batch=config.overfit_to_one_batch,
            shard=shard,
            vae_encode_fn=vae_encode_fn,
            preprocess_fn=preprocess_fn,
            additional_embedding=config.data.additional_embedding
        )
        
        eval_source, eval_target = horse2zebra(
            split="test",
            batch_size=config.training.batch_size,
            overfit_to_one_batch=config.overfit_to_one_batch,
            shard=shard,
            vae_encode_fn=vae_encode_fn,
            preprocess_fn=preprocess_fn,
            additional_embedding=config.data.additional_embedding
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
    else:
        raise ValueError(f"Unknown target dataset {config.target_dataset}")

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
    additional_embedding : Optional[str] = None, 
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

    N = min(512*16, source_labels.shape[0], target_labels.shape[0])
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
    additional_embedding : Optional[str] = None, 
    nsamples: Optional[int] = None,
) -> Tuple[dict[str,np.ndarray], dict[str,np.ndarray], np.ndarray, np.ndarray]:
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
        source_embedding = compute_embedding(source_data, embedding=additional_embedding)
        target_embedding = compute_embedding(target_data, embedding=additional_embedding)

    if vae_encode_fn is not None:
        logging.info("Preprocessing for VAE embedding.")
        source_data = [preprocess_fn(image).numpy() for image in source_data]
        target_data = [preprocess_fn(image).numpy() for image in target_data]
        logging.info("Precomputing VAE embedding.")
        source_data = compute_vae_encoding(source_data, vae_encode_fn=vae_encode_fn, batch_size=batch_size, shard=shard)
        target_data = compute_vae_encoding(target_data, vae_encode_fn=vae_encode_fn, batch_size=batch_size, shard=shard)
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
    additional_embedding : Optional[str] = None, 
    nsamples: Optional[int] = None,
) -> Tuple[dict[str,np.ndarray], dict[str,np.ndarray], np.ndarray, np.ndarray]:
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
        source_filenames = glob.glob(os.path.join(data_dir, 'trainA', '*'))
        target_filenames = glob.glob(os.path.join(data_dir, 'trainB', '*'))
    elif split == "test":
        source_filenames = glob.glob(os.path.join(data_dir, 'testA', '*'))
        target_filenames = glob.glob(os.path.join(data_dir, 'testB', '*'))
    elif split == "full":
        source_filenames = glob.glob(os.path.join(data_dir, 'trainA', '*'))
        target_filenames = glob.glob(os.path.join(data_dir, 'trainB', '*'))

        source_filenames += glob.glob(os.path.join(data_dir, 'testA', '*'))
        target_filenames += glob.glob(os.path.join(data_dir, 'testB', '*'))
    

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
        source_embedding = compute_embedding(source_data, embedding=additional_embedding)
        target_embedding = compute_embedding(target_data, embedding=additional_embedding)

    if vae_encode_fn is not None:
        logging.info("Preprocessing for VAE embedding.")
        source_data = [preprocess_fn(image).numpy() for image in source_data]
        target_data = [preprocess_fn(image).numpy() for image in target_data]
        logging.info("Precomputing VAE embedding.")
        source_data_vae = compute_vae_encoding(source_data, vae_encode_fn=vae_encode_fn, batch_size=batch_size, shard=shard)
        target_data_vae = compute_vae_encoding(target_data, vae_encode_fn=vae_encode_fn, batch_size=batch_size, shard=shard)

        source = EasyDict(data=source_data_vae)#, original_data=np.array(source_data))
        target = EasyDict(data=target_data_vae)#, original_data=np.array(target_data))

    else:
        source_data = np.array(source_data)
        target_data = np.array(target_data)

        source = EasyDict(data=source_data)
        target = EasyDict(data=target_data)

    if additional_embedding: 
        source["embedding"] = source_embedding
        target["embedding"] = target_embedding

    return source, target


def compute_vae_encoding(data : list[np.ndarray],
                         vae_encode_fn : Callable,         
                         batch_size: int,
                         shard: Optional[jax.sharding.Sharding] = None,
                         ) -> Tuple[np.ndarray, np.ndarray]:
    
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


def compute_embedding(data : list[np.ndarray],
                         embedding : str):
    if embedding == "clip":
        encode_img_fn, encode_text_fn = get_clip_fns()
        embedded_data = encode_img_fn(data)
    else:
        raise ValueError(f'embedding {embedding} is not valid.')
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
    return GenerationSampler(jnp.array(train_data), config.training.batch_size)


def cifar10(split: str) -> np.ndarray:
    """Load cifar10 data from tensorflow datasets."""
    [x_train, y_train], [x_test, y_test] = tf.keras.datasets.cifar10.load_data()
    if split == "train":
        return x_train
    else:
        return x_test
