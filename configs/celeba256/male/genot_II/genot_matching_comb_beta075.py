import ml_collections

from configs.base_uotfm import get_uotfm_config
from configs.celeba256.base_unet import get_unet_config
from configs.celeba256.base_celeba import get_celeba_config
from configs.celeba256.male.base_male import get_male_config


def get_config():
    config = get_uotfm_config()
    config = get_unet_config(config)
    config = get_celeba_config(config)
    config = get_male_config(config)

    config.training.num_steps = 400_000
    
    
    config.training.tau_a = 0.95
    config.training.tau_b = 0.95
    combine_beta = 0.75

    config.name = f"celeba256-male-genot-otclip+euclidean-beta{combine_beta}-CondVAE"
    config.wandb_group = "genot"
    config.training.is_genot = True


    config.model.cross_attn_resolutions = [i for i in range(200)]
    config.model.cross_attn_dim = config.model.input_shape[0]


    config.data.additional_embedding = "clip"
    config.model.film_cond_dim = 512
    
    # config.model.film_resolutions_down = [i for i in range(200)] # This could be 4, 8, 16, 32 
    # config.model.film_resolutions_up = [i for i in range(200)]   # This could be 4, 8, 16, 32
    config.model.film_down = [True, True, True, True] 
    config.model.film_up = [True, True, True, True, True]
    config.model.film_middle = [True, True]


    # Use this for matching
    config.training.ot_geometry = "pointcloud_combine"
    config.training.geometry_cost_matrix_kwargs =  geometry_cost_matrix_kwargs = ml_collections.ConfigDict()
    geometry_cost_matrix_kwargs.combine0 = "embedding"
    geometry_cost_matrix_kwargs.combine1 = "data"
    geometry_cost_matrix_kwargs.cost_fn0 = "cosine"
    geometry_cost_matrix_kwargs.cost_fn1 = "sqeuclidean"
    geometry_cost_matrix_kwargs.combine_beta = combine_beta

    # It won't use this!
    config.training.compare_on = "embedding"
    config.training.ot_cost_fn = "cosine"

    return config
