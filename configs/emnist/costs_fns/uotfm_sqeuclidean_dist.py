from configs.base_uotfm import get_uotfm_config
from configs.emnist.base_emnist_letters import get_emnist_letters_config
from configs.emnist.base_mlpmixer import get_mlpmixer_config


def get_config():
    ot_cost_fn = "sqeuclidean"

    config = get_uotfm_config()
    config = get_mlpmixer_config(config)
    config = get_emnist_letters_config(config)
    config.name = f"dist-fm_emnist_letters_{ot_cost_fn}-topkBOTH"
    
    config.training.matching_method = "abs_dist"
    # config.training.tau_a = 0.9
    # config.training.tau_b = 1.0
    
    config.training.ot_cost_fn = ot_cost_fn
    config.wandb_group = "costs_fns"

    
                

    return config
