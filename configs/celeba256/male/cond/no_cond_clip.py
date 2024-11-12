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

    config.name = "celeba256-male-no_cond-softmax-clip-Top4"

    config.wandb_group = "cond_II"
    config.training.cond = False

    # conditioning
    config.model.cross_attn_resolutions = []
    config.model.cross_attn_dim = 0 # config.model.input_shape[0]

    config.data.additional_embedding = "clip"
    config.model.film_resolutions = []
    config.model.film_cond_dim = 512

    config.training.num_steps = 200_000
    config.training.eval_freq = 50_000
    config.training.print_freq = 1000

    # Ot distance
    config.training.matching_method = "softmax_dist"
    config.training.compare_on = "embedding"
    config.training.ot_cost_fn = "dot"

    return config
