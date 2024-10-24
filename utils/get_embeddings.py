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
    tokenizer = model_dict["tokenizer"]
    device = model.device

    input_text = tokenizer(text, return_tensors="pt").to(device=device)

    with torch.no_grad():
        embedding = model_text(**input_text, output_hidden_states=True).hidden_states[-1].mean(dim=-2)
    
    return embedding

def get_image_embedding(model_dict, image, pooling: str = "pooler"):
    processor = model_dict["processor"]
    model = model_dict["model"]
    device = model_dict["model"].device

    input_image = processor(images=image, return_tensors="pt", padding=True).to(device=device)

    with torch.no_grad():
        embedding = model_image(**input_image_1).pooler_output

    return embedding