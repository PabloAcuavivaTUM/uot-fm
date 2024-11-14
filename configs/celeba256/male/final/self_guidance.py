from configs.base_uotfm import get_uotfm_config
from configs.celeba256.base_unet import get_unet_config
from configs.celeba256.base_celeba import get_celeba_config
from configs.celeba256.male.base_male import get_male_config


def get_config():
    config = get_uotfm_config()
    config = get_unet_config(config)
    config = get_celeba_config(config)
    config = get_male_config(config)

    config.training.num_steps = 600_000
    config.training.eval_freq_points = [10_000, 20_000, 30_000, 40_000, 75_000]
    config.training.save_all_steps = True 
    
    config.training.tau_a = 0.95
    config.training.tau_b = 0.95

    config.name = f"celeba256-male-uot-fm"
    config.wandb_group = "uot-guidance"
    config.training.is_genot = False
    config.training.genot.noise = "gaussian"

    # config.model.cross_attn_resolutions = [i for i in range(512)]
    # config.model.cross_attn_dim = config.model.input_shape[0]

    # config.data.additional_embedding = "clip"
    config.model.film_cond_dim = 512
    
    # config.model.film_resolutions_down = [i for i in range(200)] # This could be 4, 8, 16, 32 
    # config.model.film_resolutions_up = [i for i in range(200)]   # This could be 4, 8, 16, 32
    # config.model.film_down = [True, True, True, True] 
    # config.model.film_up = [True, True, True, True, True]
    # config.model.film_middle = [True, True]

    config.training.compare_on = "data"
    config.training.ot_cost_fn = "sqeuclidean"

    

    return config
