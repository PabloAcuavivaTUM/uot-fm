from configs.base_uotfm import get_uotfm_config
from configs.celeba256.add_glasses.base_glasses import get_glasses_config
from configs.celeba256.base_celeba import get_celeba_config
from configs.celeba256.base_unet import get_unet_config


def get_config():
    config = get_uotfm_config()
    config = get_unet_config(config)
    config = get_celeba_config(config)
    config = get_glasses_config(config)
    sigma = 0.3

    config.training.tau_a = 0.95
    config.training.tau_b = 0.95

    # config.overfit_to_one_batch = True
    config.name = f"celeba256-add-glasses-Attention-Normal-fromVAE-sigma{sigma}"
    config.wandb_group = "cond"
    config.training.cond = True
    config.training.cond_method = 'attention'; 
    
    #config.training.eval_freq = 5000
    # Do not modify attention (so it keeps middle and at resolution 16)
    config.training.flow_sigma = sigma
    

    return config
