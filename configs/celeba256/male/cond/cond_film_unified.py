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

    config.name = f"celeba256-male-FiLM-Unified-R*"
    config.wandb_group = "cond_II"

    # conditioning
    config.model.cross_attn_resolutions = []
    config.model.cross_attn_dim = 0 # config.model.input_shape[0]

    config.data.additional_embedding = "clip"
    config.model.film_resolutions = [i for i in range(200)]
    config.model.film_cond_dim = 512

    config.training.num_steps = 200_000
    config.training.eval_freq = 50_000
    config.training.print_freq = 1000
    ####
    # Try to reduce fitting power
    # config.model.hidden_size = 96
    # config.model.dim_mults = [2, 2, 2]

    #config.overfit_to_one_batch = True
    #config.training.num_steps = 2
    #config.training.eval_freq = 1
    #config.training.print_freq = 1
    #config.eval.num_save_samples = 3


    return config
