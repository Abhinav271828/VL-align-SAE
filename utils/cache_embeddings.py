from .model_init import init_subject_model
from .get_embeddings import get_image_embedding, get_text_embedding
from PIL import Image
import h5py
import torch
import tqdm

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
    dataset_split,
    text_model_name: str,
    image_model_name: str,
    output_paths: tuple[str, str],  # image file and text file
    device: str = "cpu",
):

    model_init_dict = init_subject_model(
        model_name=image_model_name, model_type='image', device=device
    )

    text_embed_path, image_embed_path = output_paths

    with h5py.File(text_embed_path, "w") as fout:
        for index, example in tqdm(enumerate(dataset_split['filename'])):

            # unpack the sample
            image = Image.open(f'/scratch/bani/.cache/datasets/downloads/extracted/70111e6614c7cec8efd278347b6207e7de5dd3b8babd21af307ade43cbc27d02/val2014/{example}')

            embed = get_image_embedding(model_init_dict, image)

            dimg = fimg.create_dataset(str(index), (768,))
            dimg[:] = embed.squeeze().cpu().numpy()

    model_init_dict = init_subject_model(
        model_name=text_model_name, model_type='text', device=device
    )

    with h5py.File(image_embed_pth, "w") as fimg:
        for index, example in tqdm(enumerate(dataset_split['sentences'])):

            # unpack the sample
            cap = example['raw']

            embed = get_text_embedding(model_init_dict, cap)

            dset = fout.create_dataset(str(index), (feature_count, ))

            dset[:] = c_tensor.squeeze().cpu().numpy()