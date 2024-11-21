def get_three_channels_config(config):
    # training
    config.training.num_steps = 300_000
    config.training.eval_freq = 25_000
    config.training.print_freq = 500
    config.training.cost = "sqeuclidean"
    
    # data
    config.data.channels = ["00_EU", "20_SP100", "12_RB1_pS807_S811"]
    config.data.shape = [3, 256, 256]

    return config
