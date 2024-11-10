import torch

def eval_batch(sae, batch, k):
    txt, img = batch

    # Match in latent space
    txt_latent = sae.encode_text(txt)
    img_latent = sae.encode_image(img)
    # [bz, k * 768]
    bz, _ = txt.shape
    k = min(k, bz)

    distances = torch.cdist(txt_latent, img_latent, p=2)
    txt_score_latent, img_score_latent = 0, 0
    top_k_txt = distances.topk(k, dim=1).indices
    for i in range(bz):
        if i in top_k_txt[i]:
            txt_score_latent += 1
    txt_score_latent /= bz

    top_k_img = distances.topk(k, dim=0).indices
    for i in range(bz):
        if i in top_k_img[:, i]:
            img_score_latent += 1
    img_score_latent /= bz


    # Match in output space
    txt_score_output, img_score_output = 0, 0
    # Text score
    image_map = sae.decode_image(txt_latent)
    distances = torch.cdist(img, image_map, p=2)
    top_k_txt = distances.topk(k, dim=1).indices
    for i in range(bz):
        if i in top_k_txt[i]:
            txt_score_output += 1
    txt_score_output /= bz

    # Image score
    text_map = sae.decode_text(img_latent)
    distances = torch.cdist(txt, text_map, p=2)
    top_k_img = distances.topk(k, dim=1).indices
    for i in range(bz):
        if i in top_k_img[i]:
            img_score_output += 1
    img_score_output /= bz

    return (txt_score_latent, img_score_latent), (txt_score_output, img_score_output)
