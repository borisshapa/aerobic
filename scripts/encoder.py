import argparse
import collections
import os.path

import arithmetic_compressor
import torch
from PIL import Image

from src import utils, models


def _configure_arg_parser() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--weights",
        type=str,
        default="weights/b2/encoder.pth",
        help="encoder nn weights",
    )
    arg_parser.add_argument("--b", type=int, default=2, help="Quantization parameter")
    arg_parser.add_argument(
        "--image",
        type=str,
        default="samples/b2/images/peppers.png",
        help="Path to png image to compress.",
    )
    arg_parser.add_argument(
        "--save-to",
        type=str,
        default="samples/b2/compressed/peppers.aerobic",
        help="Path to the file where to save the compressed image.",
    )
    return arg_parser


def main(args: argparse.Namespace):
    model = models.ResNetEncoder(*models.get_configs("resnet18"))
    model.load_state_dict(torch.load(args.weights, map_location=torch.device("cpu")))
    model.eval()

    img = Image.open(args.image)
    img_size = img.size
    img_tensor = utils.TRANSFORM(img)

    latent_vector = model(img_tensor.unsqueeze(0))[0].detach().cpu().tolist()
    vector_0_1, max_value = utils.to_0_1(latent_vector)
    quantized_vector = utils.quantize(vector_0_1, args.b)

    coder = arithmetic_compressor.AECompressor(
        arithmetic_compressor.models.StaticModel(utils.get_uniform_statistics(args.b))
    )
    compressed = coder.compress(quantized_vector)

    bit_string = "1" + "".join(map(str, compressed))
    result = (
        f"{max_value}:{img_size[0]}:{img_size[1]}\n".encode()
        + utils.bitstring_to_bytes(bit_string)
    )

    dir = os.path.dirname(args.save_to)
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(args.save_to, "wb+") as binary_file:
        binary_file.write(result)


if __name__ == "__main__":
    args = _configure_arg_parser().parse_args()
    main(args)
