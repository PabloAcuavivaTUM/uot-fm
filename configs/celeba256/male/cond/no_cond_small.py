from configs.base_uotfm import get_uotfm_config
from configs.celeba256.base_unet import get_unet_config
from configs.celeba256.base_celeba import get_celeba_config
from configs.celeba256.male.base_male import get_male_config


def get_config():
    config = get_uotfm_config()
    config = get_unet_config(config)
    config = get_celeba_config(config)
    config = get_male_config(config)
    
    config.training.tau_a = 0.95
    config.training.tau_b = 0.95

    # config.training.eval_freq = 5000

    config.name = "celeba256-male-no_cond-smaller"
    config.wandb_group = "cond_II"
    config.training.cond = False

    config.model.hidden_size = 64
    config.model.dim_mults = [2, 2, 4]


    return config
