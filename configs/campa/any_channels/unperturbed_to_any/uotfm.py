from configs.base_uotfm import get_uotfm_config
from configs.campa.base_unet import get_unet_config
from configs.campa.base_campa import get_campa_config
from configs.campa.any_channels.base_any_channels import get_any_channels_config
from configs.campa.any_channels.unperturbed_to_any.base_unperturbed_to_any import get_unperturbed_to_any_config

def get_config():
    config = get_uotfm_config()
    config = get_unet_config(config)
    config = get_campa_config(config)
    config = get_any_channels_config(config)
    config = get_unperturbed_to_any_config(config)

    config.name = "uot-fm-campa"
    config.training.tau_a = 0.95
    config.training.tau_b = 0.95

    return config
