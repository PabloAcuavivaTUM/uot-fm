import ml_collections


def get_unet_config(config):
    config.training.batch_size = 256
    # model
    config.model = model = ml_collections.ConfigDict()
    model.type = "unet"
    model.hidden_size = 128
    model.dim_mults = [2, 2, 2]
    model.num_res_blocks = 4
    model.heads = 1
    model.dim_head = 64
    model.attention_resolution = [16]
    model.dropout = 0.1
    model.biggan_sample = False
    model.use_vae = True
    model.input_shape = [4, 32, 32]

    # conditioning
    model.cross_attn_resolutions = []
    model.cross_attn_dim = model.input_shape[0]

    model.film_resolutions_down = []
    model.film_resolutions_up = []
    model.film_down = [False, False, False, False] 
    model.film_up = [False, False, False, False, False]
    model.film_middle = [False, False]
    model.film_cond_dim = 0

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = "adam"
    optim.learning_rate = 1e-4
    optim.beta_one = 0.9
    optim.beta_two = 0.999
    optim.eps = 1e-8
    optim.grad_clip = 1.0
    optim.warmup = 0.0
    optim.schedule = "constant"
    optim.ema_decay = 0.9999

    return config
