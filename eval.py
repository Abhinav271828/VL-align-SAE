import argparse
from sae import SAE
from torch.utils.data import DataLoader
from data import MSCOCO
from tqdm import tqdm
from utils.eval import eval_batch
import json

parser = argparse.ArgumentParser()
parser.add_argument("--model",  help="Sparsity level",
                      type=str, default='sae.pth',  required=False)
parser.add_argument("--batch",  help="Partition size",
                      type=int, default=100,  required=False)
args = parser.parse_args()

train_dl = DataLoader(MSCOCO(split='train', text_model='openai-community/openai-gpt'), batch_size=args.batch)
val_dl = DataLoader(MSCOCO(split='validation', text_model='openai-community/openai-gpt'), batch_size=args.batch)
test_dl = DataLoader(MSCOCO(split='test', text_model='openai-community/openai-gpt'), batch_size=args.batch)

cfg = json.load(open('sae_config.json'))
txt_dim = train_dl.dataset.txt_dim
img_dim = train_dl.dataset.img_dim
proj_dim = max(txt_dim, img_dim)
sae = SAE(txt_dim, img_dim, proj_dim, cfg['exp'] * proj_dim, cfg['k'])
sae.load_state_dict(torch.load('sae.pth'))

for DL in [train_dl, val_dl, test_dl]:
    num_partitions = len(DL)
    avg_txt_accuracy_per_partition = 0
    avg_img_accuracy_per_partition = 0
    for batch in tqdm(DL):
        txt_score, img_score = eval_batch(sae, batch)
        avg_txt_accuracy_per_partition += txt_score.item()
        avg_img_accuracy_per_partition += img_score.item()
    
    avg_txt_accuracy_per_partition /= len(DL)
    avg_img_accuracy_per_partition /= len(DL)

    print("Average text accuracy",  avg_txt_accuracy_per_partition)
    print("Average image accuracy", avg_img_accuracy_per_partition)