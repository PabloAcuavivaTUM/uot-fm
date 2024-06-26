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

    config.name = "celeba256-add-glasses-sqeuclidean-film-freq"
    config.wandb_group = "genot"
    config.training.is_genot = True
    config.training.eval_freq = 5000

    config.data.additional_embedding = "clip"
    config.model.film_resolutions = [i for i in range(200)]
    config.model.film_cond_dim = 512
    
    # config.training.matching_method = "softmax_dist"
    # config.training.compare_on = "embedding"
    # config.training.ot_cost_fn = "dot"

    # config.model.cross_attn_resolutions = [i for i in range(200)]
    # config.model.cross_attn_dim = config.model.input_shape[0]
    
    return config