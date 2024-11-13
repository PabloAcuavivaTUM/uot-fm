from configs.base_uotfm import get_uotfm_config
from configs.campa.base_unet import get_unet_config
from configs.campa.base_campa import get_campa_config
from configs.campa.three_channels.base_three_channels import get_three_channels_config
from configs.campa.three_channels.unperturbed_meayamycin.base_unperturbed_meayamycin import get_unperturbed_meayamycin_config

def get_config():
    config = get_uotfm_config()
    config = get_unet_config(config)
    config = get_campa_config(config)
    config = get_three_channels_config(config)
    config = get_unperturbed_meayamycin_config(config)

    config.name = "uot-fm_campa_three_channels"
    config.training.tau_a = 0.95
    config.training.tau_b = 0.95

    return config
