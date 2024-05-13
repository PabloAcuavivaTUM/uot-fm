import ml_collections


def get_celeba_config(config):
    config.task = "translation"
    config.training.gamma = "constant"
    config.training.flow_sigma = 0.01
    config.training.ot_cost_fn = "sqeuclidean"
    config.training.ot_geometry = "pointcloud"
    config.training.geometry_cost_matrix_kwargs = None
    # data
    config.data = data = ml_collections.ConfigDict()
    data.source = "celeba_attribute"
    data.target = "celeba_attribute"
    data.shape = [3, 313, 256]
    data.shuffle_buffer = 10_000
    data.random_crop = True
    data.crop_shape = [256, 256, 3]

    return config
