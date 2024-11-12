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

    config.name = "celeba256-add-glasses-TestRun"
    config.wandb_group = "ignore"

    # config.training.is_genot = True 
    ########
    config.overfit_to_one_batch = True
    config.training.num_steps = 2
    config.training.eval_freq = 1
    config.training.print_freq = 1
    config.eval.num_save_samples = 3

    return config
