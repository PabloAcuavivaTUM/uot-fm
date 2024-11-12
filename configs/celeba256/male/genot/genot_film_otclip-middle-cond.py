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

    config.training.tau_a = 0.95
    config.training.tau_b = 0.95

    config.name = "celeba256-male-genot-otclip-middlecond"
    config.wandb_group = "genot"
    config.training.is_genot = True

    config.data.additional_embedding = "clip"
    config.model.film_cond_dim = 512
    
    config.model.film_resolutions_down = []
    config.model.film_resolutions_up = []
    config.model.film_down = [False, False, False, False] 
    config.model.film_up = [False, False, False, False, False]
    config.model.film_middle = [True, True]
    

    config.training.compare_on = "embedding"
    config.training.ot_cost_fn = "cosine"

    return config
