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

    # conditioning
    config.model.cross_attn_resolutions = []
    config.model.cross_attn_dim = 0 # config.model.input_shape[0]

    # config.data.additional_embedding = "clip"
    # Make FiLM conditioning more customizable
    # config.overfit_to_one_batch = True


    config.name = f"celeba256-add-glasses-NoCond"
    config.wandb_group = "cond_film_architecture"

    return config
