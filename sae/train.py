from .model import SAE
from data import MSCOCO
from torch.utils.data import DataLoader
from random import randint
import torch
from torch import nn
import wandb
import json
from utils.eval import eval_batch
from tqdm import tqdm

def step(sample, sae, criterion):
    txt, img = sample
    txt = txt.to('cuda:0')
    img = img.to('cuda:0')

    txt_proj, text = sae.encode_text(txt)
    img_proj, image = sae.encode_image(img)

    r = randint(0, 1)
    if r:
        txt_recon = sae.decode_text(text)
        img_recon = sae.decode_image(image)

        txt_loss = criterion(txt_proj, txt_recon)
        img_loss = criterion(img_proj, img_recon)

    else:
        txt_recon = sae.decode_text(image)
        img_recon = sae.decode_image(text)

        txt_loss = criterion(txt_proj, txt_recon)
        img_loss = criterion(img_proj, img_recon)
    
    return r, txt_loss, img_loss


def train(args):
    print("Loading train dataset.")
    train = DataLoader(MSCOCO(split='train', text_model='openai-community/openai-gpt'), batch_size=args.batch)
    print("Loading validation dataset.")
    val = DataLoader(MSCOCO(split='validation', text_model='openai-community/openai-gpt'), batch_size=args.batch)

    txt_dim = train.dataset.txt_dim
    img_dim = train.dataset.img_dim
    proj_dim = max(txt_dim, img_dim)
    sae = SAE(txt_dim, img_dim, proj_dim, args.exp * proj_dim, args.k)
    sae = sae.to('cuda:0')

    criterion = nn.MSELoss()
    optim = torch.optim.Adam(sae.parameters(), lr=args.lr)

    wandb.init(project="vl-align-sae")
    wandb.run.name = wandb.run.id
    wandb.run.save()
    wandb.config.update(vars(args))

    for e in range(args.epochs):
        avg_txt_loss, avg_img_loss, avg_txt_loss_rev, avg_img_loss_rev = 0, 0, 0, 0
        avg_loss, avg_loss_rev = 0, 0
        sae.train()
        for i, sample in tqdm(enumerate(train)):
            optim.zero_grad()

            r, txt_loss, img_loss = step(sample, sae, criterion)

            loss = txt_loss + img_loss
            loss.backward()
            optim.step()
            
            if r:
                avg_txt_loss += txt_loss.item()
                avg_img_loss += img_loss.item()
                avg_loss += loss.item()
            else:
                avg_img_loss_rev += img_loss.item()
                avg_txt_loss_rev += txt_loss.item()
                avg_loss_rev += loss.item()

            if i % args.log_interval == 0:
                wandb.log({"step_txt_loss": txt_loss.item(),
                           "step_img_loss": img_loss.item()} if r else \
                          {"step_txt_loss_rev": txt_loss.item(),
                           "step_img_loss_rev": img_loss.item()})
            
        avg_txt_loss /= (len(train)/2)
        avg_img_loss /= (len(train)/2)
        avg_loss /= (len(train)/2)
        avg_txt_loss_rev /= (len(train)/2)
        avg_img_loss_rev /= (len(train)/2)
        avg_loss_rev /= (len(train)/2)

        wandb.log({"avg_txt_loss": avg_txt_loss,
                   "avg_img_loss": avg_img_loss,
                   "avg_loss": avg_loss,
                   "avg_txt_loss_rev": avg_txt_loss_rev,
                   "avg_img_loss_rev": avg_img_loss_rev,
                   "avg_loss_rev": avg_loss_rev})

        if e % args.val_interval == 0:
            avg_txt_loss, avg_img_loss, avg_txt_loss_rev, avg_img_loss_rev = 0, 0, 0, 0
            avg_loss, avg_loss_rev = 0, 0
            avg_txt_accuracy_per_partition, avg_img_accuracy_per_partition = 0, 0
            sae.eval()
            for txt, img in val:
                r, txt_loss, img_loss = step((txt, img), sae, criterion)
                if r:
                    avg_txt_loss += txt_loss.item()
                    avg_img_loss += img_loss.item()
                    avg_loss += loss.item()
                else:
                    avg_img_loss_rev += img_loss.item()
                    avg_txt_loss_rev += txt_loss.item()
                    avg_loss_rev += loss.item()

                txt = txt.to('cuda:0')
                img = img.to('cuda:0')
                
                txt_score, img_score = eval_batch(sae, (txt, img))
                avg_txt_accuracy_per_partition += txt_score.item()
                avg_img_accuracy_per_partition += img_score.item()

                if i % args.log_interval == 0:
                    wandb.log({"val_step_txt_loss": txt_loss.item(),
                               "val_step_img_loss": img_loss.item()} if r else \
                              {"val_step_txt_loss_rev": txt_loss.item(),
                               "val_step_img_loss_rev": img_loss.item()})

            avg_txt_loss /= (len(val)/2)
            avg_img_loss /= (len(val)/2)
            avg_loss /= (len(val)/2)
            avg_txt_loss_rev /= (len(val)/2)
            avg_img_loss_rev /= (len(val)/2)
            avg_loss_rev /= (len(val)/2)

            avg_txt_accuracy_per_partition /= len(val)
            avg_img_accuracy_per_partition /= len(val)

            wandb.log({"val_avg_txt_loss": avg_txt_loss,
                       "val_avg_img_loss": avg_img_loss,
                       "val_avg_loss": avg_loss,
                       "val_avg_txt_loss_rev": avg_txt_loss_rev,
                       "val_avg_img_loss_rev": avg_img_loss_rev,
                       "val_avg_loss_rev": avg_loss_rev,
                       "val_avg_txt_accuracy": avg_txt_accuracy_per_partition,
                       "val_avg_img_accuracy": avg_img_accuracy_per_partition})


    i = 0
    torch.save(sae.state_dict(), (f'sae_{i}.pth'))
    torch.save(model.state_dict(), os.path.join(dir_name, 'model.pth'))
    with open(os.path.join(dir_name, f'config_{i}.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    wandb.finish()