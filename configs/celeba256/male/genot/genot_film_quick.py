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

    config.name = "celeba256-male-genot-quick-test-film"
    config.wandb_group = "genot"
    config.training.is_genot = True

    config.data.additional_embedding = "clip"
    config.model.film_resolutions = [i for i in range(200)]
    config.model.film_cond_dim = 512

    config.overfit_to_one_batch = True
    config.training.num_steps = 200
    config.training.eval_freq = 10
    config.training.print_freq = 5
    config.eval.num_save_samples = 3


    config.model.hidden_size = 8
    config.model.dim_mults = [1, 1, 1]


    return config
