import torch

def eval_batch(sae, batch):
    txt, img = batch
    _, txt_latent = sae.encode_text(txt)
    _, img_latent = sae.encode_image(img)
    # [bz, k * 768]
    bz, _ = txt.shape

    distances = torch.cdist(txt_latent, img_latent, p=2)
    txt_score = (distances.argmax(dim=1) == torch.arange(bz, device='cuda:0')).sum() / bz
    img_score = (distances.argmax(dim=0) == torch.arange(bz, device='cuda:0')).sum() / bz
    return txt_score, img_score
