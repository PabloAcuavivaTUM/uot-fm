from configs.campa.any_channels.unperturbed_to_any.uotfm import get_config as base_uotfm_cfg
from itertools import chain
####
# OBJECTIVE: This with *_channels1 -> To checkk if learning the mapping by channels and learning it together 
# gives difference in accuracy
####

def extend_features(features, extension_dict):
    # We need it to properly deal with named features which generate multiple features
    updated_features = []
    for feature in features:
        if feature in extension_dict:
            updated_features.extend([f"{feature}_{i}" for i in range(extension_dict[feature])])
        else:
            updated_features.append(feature)
    return updated_features


bio_channels = {
    "184A1_meayamycin": [
        "10_POL2RA_pS2",
        "01_PABPC1",
        "09_SRRM2",
    ],
    "184A1_CX5461": [
        "21_NCL",
        "03_RPS6",
        "10_H3K27ac",
    ],
    "184A1_AZD4573": [
        "13_POL2RA_pS5",
        "03_CDK9",
        "09_SRRM2",
    ],
    "184A1_triptolide": [
        "10_POL2RA_pS2",
        "13_PABPN1",
        "08_H3K4me3",
    ],
    "184A1_TSA": [
        "10_H3K27ac",
        "17_HDAC3",
        "14_PCNA",
    ]
}

def get_config():
    config = base_uotfm_cfg()
    type_tgt = '184A1_triptolide'
    config.hacky_embedding_before_vae = False 
    config.name = f"bio_{type_tgt}_base_genot_vae_after"
    config.wandb_group = "slides_campa"
    config.training.is_genot = True
    config.training.genot.noise = "gaussian"
    config.training.eval_freq_points = [100, 10_000, 15_000, 20_000, 30_000, 35_000, 40_000] 
    config.training.num_steps = 400_000
    
    config.data.type_tgt=[type_tgt] # , "184A1_CX5461", "184A1_triptolide"]


    # Make sure if fits into 1 GPU
    config.training.batch_size = 128
    config.training.batch_size_matching = 256


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

    channels0 = bio_channels[type_tgt]
    grouped_channels = [channels0] # , channels1]
    channels = list(chain.from_iterable(grouped_channels))

    config.eval.checkpoint_metric = f'{type_tgt}.[channel_umap__{channels0[0]}]-FID-target'

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
