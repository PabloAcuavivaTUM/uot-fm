import ml_collections


def get_horse2zebra_config(config):
    config.task = "translation"
    config.training.gamma = "constant"
    config.training.flow_sigma = 0.01
    config.training.ot_cost_fn = "sqeuclidean"
    config.training.ot_geometry = "pointcloud"
    config.training.geometry_cost_matrix_kwargs = None
    
    # data
    config.data = data = ml_collections.ConfigDict()
    data.source = "horse2zebra"
    data.target = "horse2zebra"
    data.shape = [3, 256, 256]
    data.shuffle_buffer = 10_000
    data.additional_embedding = None

    
    
    
    return config
