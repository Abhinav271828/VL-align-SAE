from .model_init import init_subject_model
from .get_embeddings import get_image_embedding, get_text_embedding
from PIL import Image
import h5py
from tqdm import tqdm
import os

def create_output_filename(
    input_path: str, output_dir: str, model_name: str, pool_strat: str = "pooler"
) -> str:
    """
    Create the output filename for the embeddings
    Inputs:
    * input_path: str: path to the input file
    * output_dir: str: path to the output directory
    * model_name: str: name of the model
    Outputs:
    (str, str): the output filename for text and image
    """
    basename = input_path.split("/")[-1].split(".")[0]
    model_name = model_name.replace("/", "_")
    return (
        f"{output_dir}/{model_name}_{basename}_{pool_strat}_text.h5",
        f"{output_dir}/{model_name}_{basename}_img.h5",
    )

def convert_raw_to_embeddings(
    split : str,
    dataset,
    text_model_name: str,
    image_model_name: str,
    output_paths: tuple[str, str],  # image file and text file
    create_image: bool, create_text: bool,
    device: str = "cpu",
):

    text_embed_path, image_embed_path = output_paths

    print("...for image...")
    if create_image:
        model_init_dict = init_subject_model(
            model_name=image_model_name, model_type='image', device='cuda:0'
        )

        with h5py.File(image_embed_path, "w") as fimg:
            for idx, example in tqdm(enumerate(dataset[split])):
                embed = get_image_embedding(model_init_dict, example['image'])

                dimg = fimg.create_dataset(str(idx), (768,))
                dimg[:] = embed.squeeze().cpu().numpy()
    else: print("already exists!")

    print("...and text.")
    if create_text:
        model_init_dict = init_subject_model(
            model_name=text_model_name, model_type='text', device='cuda:0'
        )

        with h5py.File(text_embed_path, "w") as ftxt:
            for idx, example in tqdm(enumerate(dataset[split])):
                cap = example['sentences']['raw']
                embed = get_text_embedding(model_init_dict, cap)

                d = embed.size(-1)
                dset = ftxt.create_dataset(str(idx), (d, ))
                dset[:] = embed.squeeze().cpu().numpy()
    else: print("already exists!")