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

    # config.overfit_to_one_batch = True
    config.name = "test-celeba256-add-glasses-cond"
    config.wandb_group = "test"
    # config.training.num_steps = 10
    config.training.cond = True

    config.data.source = "celeba_fake"
    config.data.target = "celeba_fake"

    # config.training.batch_size = 256

    return config
