from transformers import AutoModelForCausalLM, OpenAIGPTLMHeadModel, AutoTokenizer, AutoImageProcessor, AutoConfig

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
            tokenizer = tokenizer = AutoTokenizer.from_pretrained(model_name)
            if 'gpt' in model_name: # openai-community/openai-gpt
                model = OpenAIGPTLMHeadModel.from_pretrained(model_name)
            elif 'llama' in model_name: # meta-llama/Llama-3.2-1B
                model = AutoModelForCausalLM.from_pretrained(model_name)

            model.to(device=device)
            model.eval()
            return {
                "model_text": model,
                "tokenizer": tokenizer,
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
