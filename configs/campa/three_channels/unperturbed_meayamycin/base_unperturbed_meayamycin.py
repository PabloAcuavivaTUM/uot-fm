def get_unperturbed_meayamycin_config(config):
    # data
    config.data.type_src="184A1_unperturbed"
    config.data.type_tgt="184A1_meayamycin"  

    config.eval.cell_embeddings_metrics = []
    

    return config
