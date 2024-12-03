from configs.campa.any_channels.unperturbed_meayamycin.uotfm import get_config as base_uotfm_cfg
from itertools import chain

def extend_features(features, extension_dict):
    # We need it to properly deal with named features which generate multiple features
    updated_features = []
    for feature in features:
        if feature in extension_dict:
            updated_features.extend([f"{feature}_{i}" for i in range(extension_dict[feature])])
        else:
            updated_features.append(feature)
    return updated_features

def get_config():
    config = base_uotfm_cfg()

    config.name = f"any_channel_base_genot_meayamycin"
    config.wandb_group = "campa"
    config.training.is_genot = True
    config.training.genot.noise = "gaussian"
    config.training.eval_freq_points = [100, 1_000, 5_000, 10_000, 15_000, 20_000] 
    # config.training.num_steps = 2 # Just to check everything is running

    # Make sure if fits into 1 GPU
    config.training.batch_size = 64
    config.training.batch_size_matching = 256


    config.eval.checkpoint_metric = '184A1_meayamycin.[channel_umap__00_EU]-FID-target'

    ####
    #
    intensity_features = ['mean_intensity',
                          'median_intensity',
                          'std_dev_intensity',
                          'max_intensity',
                          'cv_intensity',
                          'skewness_intensity',
                          'kurtosis_intensity',
                          'energy_intensity',
                          # 'entropy_intensity',
                          'mad_intensity',
                        ]

    morphological_features = ['area',
                              'perimeter',
                              'eccentricity',
                              'solidity',
                              'major_axis_length',
                              'minor_axis_length',
                              'orientation',
                              'circularity',
                              'convex_area',
                              'extent',
                              'equivalent_diameter',
                              'bbox',
                              'centroid',
                              'filled_area',
                              'aspect_ratio',
                            ]
    morphological_features_extension = {'bbox': 4, 'centroid': 2}

    channels0 = ["00_EU", "20_SP100", "12_RB1_pS807_S811"]
    channels1 = ["07_H2B","15_U2SNRNPB", "20_ALYREF"]
    grouped_channels = [channels0, channels1]
    channels = list(chain.from_iterable(grouped_channels))


    config.data.additional_embedding = {
      "morphological_features": dict(features_list=morphological_features, random_state=42),
    }
    for channel in channels:
        config.data.additional_embedding[f"channel_umap__{channel}"] = dict(n_components=16, random_state=42)
        config.data.additional_embedding[f"channel_features__{channel}"] = dict(features_list=intensity_features)


    
    config.data.embedding_combinations = {"embedding": ["morphological_features"]
                                         + [f"channel_umap__{channel}" for channel in channels]
                                        } 
    # ! Must have the same dimension as "embedding"
    config.model.film_cond_dim = 19 + 16 * len(channels)

    
    # Where to compare
    config.training.compare_on = "morphological_features"    
    
    config.eval.cell_embeddings_metrics = [
        "morphological_features", 
    ] + [f"channel_umap__{channel}" for channel in channels]

    config.eval.cell_embeddings_histograms = {
        "morphological_features": extend_features(morphological_features, morphological_features_extension),
    }
    for channel in channels:
        config.eval.cell_embeddings_histograms[f"channel_features__{channel}"] = intensity_features


    #### 
    config.model.vae_fns = "naive_concat"
    config.model.input_shape = [4*len(grouped_channels), 32, 32]
    config.data.channels = channels
    config.eval.image_channels = grouped_channels
    config.data.shape = [3*len(grouped_channels), 256, 256]


    ####
    # Configuration of FiLM layers 
    # This could be 4, 8, 16, 32
    config.model.film_resolutions_down = [i for i in range(200)]  
    config.model.film_resolutions_up = [i for i in range(200)]  
    config.model.film_down = [True, True, True, True] 
    config.model.film_up = [True, True, True, True, True]
    config.model.film_middle = [True, True]


    config.training.ot_cost_fn = "sqeuclidean"

    

    return config
