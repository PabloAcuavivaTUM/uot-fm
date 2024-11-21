from typing import Any, List, Callable

from skimage import filters, measure, morphology
from scipy.stats import skew, kurtosis
import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def to_dataframe(fdict: str) -> pd.DataFrame:
    data = {}
    for k, v in fdict.items():
        if isinstance(v, dict):
            nested_df = to_dataframe(v)
            for nested_k, nested_v in nested_df.items():
                data[f"{k}_{nested_k}"] = nested_v
        else:
            arr = np.array(v).reshape(-1)
            if len(arr) == 1:
                data[k] = arr
            else:
                for i in range(len(arr)):
                    data[f"{k}_{i}"] = arr[i : i + 1]

    return pd.DataFrame(data)


def to_array(fdict: dict) -> np.ndarray:
    arrs = []
    for k, v in fdict.items():
        if isinstance(v, dict):
            arrs += [to_array(v)]
        else:
            arrs += [np.array(v).reshape(-1)]

    return np.concatenate(arrs)


class FeaturesDict(dict):
    """Convenience class for managing Cell Features."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

    def __dir__(self):
        return list(self.keys()) + super().__dir__()

    @classmethod
    def stack(cls, featuresdict_list: list["FeaturesDict"]) -> "FeaturesDict":
        stacked_features_dict = cls()
        for k in featuresdict_list[0]:
            stacked_features_dict[k] = np.stack([featuresdict[k] for featuresdict in featuresdict_list])

        return stacked_features_dict

    def unstack(self) -> List["FeaturesDict"]:
        n0 = len(next(iter(self.values())))
        for v in self.values():
            if len(v) != n0:
                raise ValueError(f"{type(self).__name__} is not well formatted so it cannot be unstacked.")

        features_unstacked = dict()
        for k, v in self.items():
            features_unstacked[k] = np.split(v, n0, axis=0)

        features_dict_list = []
        for i in range(n0):
            feature_dict_i = FeaturesDict()
            for k in self:
                feature_dict_i[k] = features_unstacked[k][i]
            features_dict_list.append(feature_dict_i)

        return features_dict_list

    def to_array(self) -> np.ndarray:
        return to_array(self)

    def to_dataframe(self) -> pd.DataFrame:
        return to_dataframe(self)


class FunctionRegistry:
    def __init__(self):
        self.registry = {}

    def register(self, func):
        """Register a function with its name automatically derived from the function."""
        func_name = func.__name__
        if func_name in self.registry:
            raise ValueError(f"Function '{func_name}' is already registered.")
        self.registry[func_name] = func
        return func

    def get_fn(self, name: str):
        """Retrieve a registered function by name."""
        if name not in self.registry:
            raise ValueError(f"Function '{name}' not found in registry.")
        return self.registry.get(name)

    def apply_fn(self, name: str, *args, **kwargs):
        """Execute a registered function by name."""
        func = self.get_fn(name)
        if func is not None:
            return func(*args, **kwargs)


###
# Intensity Registries (They work on the channels)
intensity_features_registry = FunctionRegistry()


@intensity_features_registry.register
def mean_intensity(flatten_channels: np.ndarray) -> np.ndarray:
    return np.nanmean(flatten_channels, axis=1)


@intensity_features_registry.register
def median_intensity(flatten_channels: np.ndarray) -> np.ndarray:
    return np.nanmedian(flatten_channels, axis=1)


@intensity_features_registry.register
def std_dev_intensity(flatten_channels: np.ndarray) -> np.ndarray:
    return np.nanstd(flatten_channels, axis=1)


@intensity_features_registry.register
def max_intensity(flatten_channels: np.ndarray) -> np.ndarray:
    return np.nanmax(flatten_channels, axis=1)


@intensity_features_registry.register
def cv_intensity(flatten_channels: np.ndarray) -> np.ndarray:
    mean_intensity = np.nanmean(flatten_channels, axis=1)
    std_dev = np.nanstd(flatten_channels, axis=1)
    return std_dev / mean_intensity


@intensity_features_registry.register
def skewness_intensity(flatten_channels: np.ndarray) -> np.ndarray:
    return skew(flatten_channels, axis=1, nan_policy="omit")


@intensity_features_registry.register
def kurtosis_intensity(flatten_channels: np.ndarray) -> np.ndarray:
    return kurtosis(flatten_channels, axis=1, nan_policy="omit")


@intensity_features_registry.register
def energy_intensity(flatten_channels: np.ndarray) -> np.ndarray:
    return np.nansum(np.square(flatten_channels), axis=1)


@intensity_features_registry.register
def entropy_intensity(flatten_channels: np.ndarray) -> np.ndarray:
    return -np.nansum(flatten_channels * np.log(flatten_channels + 1e-10), axis=1)


@intensity_features_registry.register
def mad_intensity(flatten_channels: np.ndarray) -> np.ndarray:
    mean_intensity = np.nanmean(flatten_channels, axis=1)
    return np.nanmean(np.abs(flatten_channels - mean_intensity[:, None]), axis=1)


all_intensity_features = list(intensity_features_registry.registry.keys())


def calculate_intensity_features(
    channels: np.ndarray,
    features_list: list[str],
    remove_zeros_from_stats: bool = True,
    drop_cell: bool = False,
    verbose: bool = False,
    j: int = 0,
) -> List[FeaturesDict]:
    ###
    # Set up & visualization

    N, h, w, c = channels.shape

    if verbose:
        plt.figure(figsize=(12, 4))
        for channel in range(c):
            img = channels[j]
            plt.subplot(1, c, channel + 1)
            plt.imshow(img[:, :, channel].astype(np.int8))
            plt.title(f"Channel {channel}")
            plt.axis("off")

    # Perform filter to remove cell
    if drop_cell:
        m = np.quantile(channels[channels != 0], 0.5) * 1.05
        channels = channels * (channels > m)

    if verbose:
        plt.figure(figsize=(12, 4))
        for channel in range(img.shape[-1]):
            img = channels[j]
            plt.subplot(1, img.shape[-1], channel + 1)
            plt.imshow(img[:, :, channel].astype(np.int8))
            plt.title(f"Channel {channel}")
            plt.axis("off")

    ###
    # Feature calculation

    flatten_channels = channels.reshape(N, -1, c)
    if remove_zeros_from_stats:
        flatten_channels = np.where(flatten_channels != 0, flatten_channels, np.nan)

    imgs_intensity_features = FeaturesDict()
    for feature in features_list:
        feature_stat = intensity_features_registry.apply_fn(feature, flatten_channels)
        feature_stat = np.where(np.isnan(feature_stat), 0, feature_stat)
        imgs_intensity_features[feature] = feature_stat

    return imgs_intensity_features.unstack()


###
# Morphological Registries (They work on the segmentation masks)
morphological_features_registry = FunctionRegistry()

all_morphological_features = [
    "area",
    "perimeter",
    "eccentricity",
    "solidity",
    "major_axis_length",
    "minor_axis_length",
    "orientation",
    "circularity",
    "convex_area",
    "extent",
    "equivalent_diameter",
    "bbox",
    "centroid",
    "filled_area",
    "aspect_ratio",
    # "central_moments",
    # "hu_moments", # ! TODO: Look into them, for now deprectaed
] + list(morphological_features_registry.registry.keys())

# TODO: not the best way to save normalizing constants, but good enough for now. Just ot have in range where matching is meaningful
morphological_features_norm_constants = {
    'area': (3.423000e+03, 6.340000e+04),
    'perimeter': (2.848122e+02, 1.770947e+03),
    'eccentricity': (8.980376e-02, 9.755505e-01),
    'solidity': (6.026649e-01, 9.962288e-01),
    'major_axis_length': (7.963098e+01, 3.473195e+02),
    'minor_axis_length': (5.039151e+01, 2.781226e+02),
    'orientation': (-1.570730e+00, 1.570705e+00),
    'circularity': (1.865850e-01, 8.002663e-01),
    'convex_area': (3.942000e+03, 6.364000e+04),
    'extent': (3.860294e-01, 9.674072e-01),
    'equivalent_diameter': (6.601741e+01, 2.841186e+02),
    'bbox': (0.0, 256),
    'centroid': (0.0, 256.0),
    'filled_area': (3.423000e+03, 6.340000e+04),
    'aspect_ratio': (1.004057e+00, 4.550102e+00)
}


def calculate_single_cell_morphological_features(segmentation_mask: np.ndarray):
    segmentation_mask = np.squeeze(segmentation_mask)

    threshold = filters.threshold_otsu(segmentation_mask)
    binary_image = segmentation_mask > threshold

    # This is probably unneccesary
    clean_binary_image = morphology.remove_small_objects(binary_image, min_size=30)

    regions = measure.regionprops(clean_binary_image.astype(int))

    if regions:
        if len(regions) != 1:
            raise ValueError(f"We found {len(regions)} possible cells... check")
        region_props = regions[0]

        # Extract  central moments
        # moments = measure.moments_central(
        #     clean_binary_image,
        #     center=(region_props.centroid[0], region_props.centroid[1]),
        #     order=4,
        # )

        #hu_moments = measure.moments_hu(moments)

        img_morphological_features = FeaturesDict(
            **{
                "area": region_props.area,
                "perimeter": region_props.perimeter,
                "eccentricity": region_props.eccentricity,
                "solidity": region_props.solidity,
                "major_axis_length": region_props.major_axis_length,
                "minor_axis_length": region_props.minor_axis_length,
                "orientation": region_props.orientation,
                "circularity": (4 * np.pi * region_props.area) / (region_props.perimeter**2),
                "convex_area": region_props.convex_area,
                "extent": region_props.extent,
                "equivalent_diameter": region_props.equivalent_diameter,
                "bbox": region_props.bbox,
                "centroid": region_props.centroid,
                "filled_area": region_props.filled_area,
                "aspect_ratio": region_props.major_axis_length / region_props.minor_axis_length,
                #"central_moments": moments,
                #"hu_moments": hu_moments,
            }
        )
    else: # Prevent errors
        img_morphological_features = FeaturesDict(
            **{
                "area": 0.0,
                "perimeter": 0.0,
                "eccentricity": 0.0,
                "solidity": 0.0,
                "major_axis_length": 0.0,
                "minor_axis_length": 0.0,
                "orientation": 0.0,
                "circularity": 0.0,
                "convex_area": 0.0,
                "extent": 0.0,
                "equivalent_diameter": 0.0,
                "bbox": [0.0,0.0,0.0,0.0],
                "centroid": [0.0,0.0],
                "filled_area": 0.0,
                "aspect_ratio": 0.0,
                #"central_moments": moments,
                #"hu_moments": hu_moments,
            }
        )

    return img_morphological_features


def calculate_morphological_features(
    segmentation_masks: np.ndarray,
    features_list: list[str],
    remove_zeros_from_stats: bool = True,
) -> List[FeaturesDict]:
    _morphological_features_list = [
        calculate_single_cell_morphological_features(segmentation_masks[i]) for i in range(len(segmentation_masks))
    ]
    morphological_features_bulk = FeaturesDict.stack(_morphological_features_list)
    img_morphological_features = FeaturesDict()

    N, *_ = segmentation_masks.shape
    flatten_segmentation_masks = segmentation_masks.reshape(N, -1)
    if remove_zeros_from_stats:
        flatten_segmentation_masks = np.where(flatten_segmentation_masks != 0, flatten_segmentation_masks, np.nan)

    for feature in features_list:
        if feature in morphological_features_bulk:
            feature_value = morphological_features_bulk[feature]
        else:
            feature_stat = morphological_features_registry.apply_fn(feature, flatten_segmentation_masks)
            feature_stat = np.where(np.isnan(feature_stat), 0, feature_stat)
            feature_value = feature_stat
        
        # Normalize feature
        _min, _max = morphological_features_norm_constants.get(feature, (0, 1))
        feature_value = (feature_value - _min) / (_max - _min)

        img_morphological_features[feature] = feature_value 

    return img_morphological_features.unstack()
