from configs.base_uotfm import get_uotfm_config
from configs.celeba256.add_glasses.base_glasses import get_glasses_config
from configs.celeba256.base_celeba import get_celeba_config
from configs.celeba256.base_unet import get_unet_config


def get_config():
    config = get_uotfm_config()
    config = get_unet_config(config)
    config = get_celeba_config(config)
    config = get_glasses_config(config)

    config.training.tau_a = 0.95
    config.training.tau_b = 0.95

    config.name = f"celeba256-add-glasses-FiLM-Unified-R16"
    config.wandb_group = "cond"

    # conditioning
    config.model.cross_attn_resolutions = []
    config.model.cross_attn_dim = 0 # config.model.input_shape[0]

    config.data.additional_embedding = "clip"
    config.model.film_resolutions = [16]
    config.model.film_cond_dim = 512
    
    return config