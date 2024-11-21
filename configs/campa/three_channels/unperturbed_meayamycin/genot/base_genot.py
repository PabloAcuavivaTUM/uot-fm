from configs.campa.three_channels.unperturbed_meayamycin.uotfm import get_config as base_uotfm_cfg


def get_config():
    config = base_uotfm_cfg()

    config.name = f"testing_three_channel_base_genot"
    config.wandb_group = "campa"
    config.training.is_genot = True
    config.training.genot.noise = "gaussian"
    config.training.eval_freq_points = [100, 1_000, 5_000, 10_000, 15_000, 20_000]
    config.training.num_steps = 200 # Just to check everything is running

    # Make sure if fits into 1 GPU
    config.training.batch_size = 64
    config.training.batch_size_matching = 256



    config.eval.checkpoint_metric = '[channel_umap__00_EU]-FID-target'

    # TEMPORAL: See what is happening
    # config.eval.num_save_samples = 4

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


    config.data.additional_embedding = {
      "morphological_features": dict(features_list=morphological_features, random_state=42),
      "channel_umap__00_EU": dict(n_components=16, random_state=42),
      "channel_umap__12_RB1_pS807_S811": dict(n_components=16, random_state=42),
      "channel_umap__20_SP100": dict(n_components=16, random_state=42),

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
    
    config.eval.cell_embeddings_metrics = [
        "morphological_features", 
        "channel_umap__00_EU", 
        "channel_umap__20_SP100",
        "channel_umap__12_RB1_pS807_S811",
    ]

    config.eval.cell_embeddings_histograms = {
        "morphological_features": morphological_features,
        "channel_features__00_EU": intensity_features,
        "channel_features__20_SP100": intensity_features,
        "channel_features__12_RB1_pS807_S811": intensity_features,
    }

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
