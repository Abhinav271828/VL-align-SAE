import torch

def get_text_embedding(model_dict, text, pooling="pooler"):
    """
    Get the text features across layers
    Inputs:
    * model_text: torch.nn.Module: the text model
    * text: str: the text
    * pooling: str: the pooling strategy
    Outputs:
    List[torch.Tensor]: the text features across layers
    """
    model = model_dict["model"]
    model_text = model_dict["model_text"]
    tokenizer = model_dict["tokenizer"]

    text_projection = model.text_projection
    device = model.device
    eos_token = model_text.config.eos_token_id

    input_text = tokenizer(text, return_tensors="pt").to(device=device)
    eos_token_index = (input_text["input_ids"] == eos_token).int().argmax(dim=-1)

    text_features_post_proj = []
    text_features_pre_proj = []

    with torch.no_grad():
        hiddens = model_text(**input_text, output_hidden_states=True).hidden_states
        if pooling == "pooler":
            text_features_pre_proj = [
                model_text.final_layer_norm(embds)[:, eos_token_index, :].squeeze()
                for embds in hiddens
            ]

            text_features_post_proj = [
                text_projection(embds).squeeze() for embds in text_features_pre_proj
            ]

        if pooling == "mean":
            text_features_pre_proj = [
                model_text.final_layer_norm(embds).mean(dim=1).squeeze()
                for embds in hiddens
            ]

            text_features_post_proj = [
                text_projection(embds).squeeze() for embds in text_features_pre_proj
            ]

    return {
        "text_features_pre_proj": text_features_pre_proj,
        "text_features_post_proj": text_features_post_proj,
    }

def get_image_embedding(model_dict, image, pooling: str = "pooler"):
    processor = model_dict["processor"]
    model = model_dict["model"]
    device = model_dict["model"].device

    input_image = processor(images=image, return_tensors="pt", padding=True).to(device=device)

    with torch.no_grad():
        embedding = model_image(**input_image_1).pooler_output

    return embedding