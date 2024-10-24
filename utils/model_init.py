from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, AutoConfig

def init_subject_model(
    model_name: str, model_type: str, model_config=None, device: str = "cpu"
) -> dict:
    """
    To initialize the subject model (the one being studied)
    Inputs:
    * model_name: str: name of the model (as in huggingface)
    * model_type: str: type of the model (Text Encoder or image)
    * model_config: Config object of the model
    Outputs:
    dict: dictionary containing the model and the config and related
    """

    match model_type:
        case "text":
            if model_config is None:
                model_config = AutoConfig.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name, config=model_config)
            model.to(device=device)
            model.eval()
            return {
                "model_text": model,
                "tokenizer": tokenizer,
                "config_text": model_config,
            }
        case "image": # facebook/dinov2-base
            config = AutoConfig.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name, config=model_config)
            model.to(device=device)
            processor = AutoImageProcessor.from_pretrained(model_name)
            model.eval()
            return {
                "model": model,
                "processor": processor,
            }
        case _:
            raise ValueError("Model type not recognized")