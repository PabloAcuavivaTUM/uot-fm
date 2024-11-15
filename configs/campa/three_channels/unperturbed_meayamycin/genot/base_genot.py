from configs.campa.three_channels.unperturbed_meayamycin.uotfm import get_config as base_uotfm_cfg


def get_config():
    config = base_uotfm_cfg()

    config.name = f"three_channel_base_genot"
    config.wandb_group = "campa"
    config.training.is_genot = True
    config.training.genot.noise = "gaussian"

    # config.model.cross_attn_resolutions = [i for i in range(512)]
    # config.model.cross_attn_dim = config.model.input_shape[0]

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
  'entropy_intensity',
  'mad_intensity']

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



    config.data.additional_embedding = {
      "morphological_umap": dict(n_components=16),
      "channel_umap__00_EU": dict(n_components=16),
      "channel_umap__20_SP100": dict(n_components=16),
      "channel_umap__12_RB1_pS807_S811": dict(n_components=16),
      "morphological_features": dict(features_list=morphological_features),
      "channel_features__00_EU": dict(features_list=intensity_features),
      "channel_features__20_SP100": dict(features_list=intensity_features),
      "channel_features__12_RB1_pS807_S811": dict(features_list=intensity_features),
    }
    
    config.data.embedding_combinations = {"embedding": ["morphological_features", 
                                                        "channel_umap__00_EU", 
                                                        "channel_umap__20_SP100",
                                                        "channel_umap__12_RB1_pS807_S811"]
                                        } 
    # ! Must have the same dimension as "embedding"
    config.model.film_cond_dim = 19 + 16 + 16 + 16
    
    # Where to compare
    config.training.compare_on = "morphological_features"    
    
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
