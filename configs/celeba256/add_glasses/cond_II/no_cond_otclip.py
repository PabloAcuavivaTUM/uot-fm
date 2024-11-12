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

    # Same config as II just adding more tranining steps
    config.name = "celeba256-add-glasses-NoCond-otclip-II"
    config.wandb_group = "cond"

    config.data.additional_embedding = "clip"
    
    # OT-Distance
    # config.training.matching_method = "softmax_dist"
    config.training.compare_on = "embedding"
    config.training.ot_cost_fn = "cosine"

    
    # Extra from I -> II
    config.training.num_steps = 200_000

    ########
    # config.overfit_to_one_batch = True
    # config.training.num_steps = 5
    # config.training.eval_freq = 1
    # config.training.print_freq = 1
    # config.eval.num_save_samples = 3

    return config
