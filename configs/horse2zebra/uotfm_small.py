from configs.base_uotfm import get_uotfm_config
from configs.horse2zebra.base_unet import get_unet_config
from configs.horse2zebra.base_horse2zebra import get_horse2zebra_config

def get_config():
    config = get_uotfm_config()
    config = get_unet_config(config)
    config = get_horse2zebra_config(config)

    # training
    config.training.num_steps = 100_000
    config.training.eval_freq = 10_000
    config.training.print_freq = 1000

    # ! This is not right not available
    config.data.map_forward = True
    # config.data.precomputed_stats_file = "horse2zebra"
    
    # horse2zebra does not have labels
    config.eval.labelwise = False
    config.data.eval_labels = None

    ######
    config.name = "uot-fm_horse2zebra32"
    config.wandb_group = "ignore"

    config.training.tau_a = 0.95
    config.training.tau_b = 0.95

    ######################################
    # Make it into small images without VAE
    config.data.shape = [3, 32, 32] 
    config.model.use_vae = False
    config.data.precomputed_stats_file = "horse2zebra32"
    config.model.input_shape = [3, 32, 32]
    config.model.num_res_blocks = 3


    return config
