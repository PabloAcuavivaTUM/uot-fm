from configs.base_uotfm import get_uotfm_config
from configs.celeba256.base_unet import get_unet_config
from configs.celeba256.base_celeba import get_celeba_config
from configs.celeba256.add_glasses.base_glasses import get_glasses_config


def get_config():
    ot_cost_fn = "l1"
    ot_geometry = "geodesic"
    
    config = get_uotfm_config()
    config = get_unet_config(config)
    config = get_celeba_config(config)
    config = get_glasses_config(config)
    config.name = f"uot-fm_celeba256_add_glasses_{ot_cost_fn}_{ot_geometry}"
    config.training.tau_a = 0.95
    config.training.tau_b = 0.95
    
    config.training.ot_cost_fn = ot_cost_fn
    config.wandb_group = "costs_fns"
    config.training.ot_geometry = ot_geometry
    

    return config
