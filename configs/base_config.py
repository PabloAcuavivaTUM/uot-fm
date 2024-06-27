import ml_collections

from warnings import warn

from configs.wandb_config import get_wandb

def get_base_config():
    config = ml_collections.ConfigDict()
    config.seed = 42
    config.overfit_to_one_batch = False
    try:
        config = get_wandb(config)
    except:
        warn('It was not possible to set up any wandb configuration. Setting all values to empty')
        config.wandb_key = ""
        config.wandb_group = ""
        config.wandb_entity = ""

    # training
    config.training = training = ml_collections.ConfigDict()
    training.print_freq = 1000
    training.save_checkpoints = True
    training.preemption_ckpt = False
    training.ckpt_freq = 10000
    training.resume_ckpt = False
    training.is_genot = False
    training.genot = genot = ml_collections.ConfigDict()
    genot.noise = 'gaussian'
    genot.x0_prob = 0.0

    config.eval = eval = ml_collections.ConfigDict()
    eval.compute_metrics = True
    eval.enable_fid = True
    eval.enable_path_lengths = True
    eval.enable_mse = False
    eval.checkpoint_metric = "fid"
    eval.save_samples = True
    eval.num_save_samples = 7
    eval.labelwise = True
    eval.checkpoint_step = None

    return config
