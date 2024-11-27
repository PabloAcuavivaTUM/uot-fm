def get_unperturbed_to_any_config(config):
    # data
    config.data.type_src="184A1_unperturbed"
    config.data.type_tgt=["184A1_meayamycin"]

    config.eval.cell_embeddings_metrics = []
    config.eval.cell_embeddings_histograms = dict()
    

    return config
