import argparse

import arithmetic_compressor
import numpy as np
import torch
from PIL import Image

from src import utils, models


def _configure_arg_parser() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--weights",
        type=str,
        default="weights/b2/decoder.pth",
        help="decoder nn weights",
    )
    arg_parser.add_argument("--b", type=int, default=2, help="Quantization parameter")
    arg_parser.add_argument(
        "--file",
        type=str,
        default="samples/b2/compressed/peppers.aerobic",
        help="Path to binary file to decompress.",
    )
    arg_parser.add_argument(
        "--save-to",
        type=str,
        default="samples/b2/decompressed/aerobic_peppers.png",
        help="Path to the file where to save the decompressed image.",
    )
    return arg_parser


def main(args: argparse.Namespace):
    model = models.ResNetDecoder(*models.get_configs("resnet18"))
    model.load_state_dict(torch.load(args.weights, map_location=torch.device("cpu")))
    model.eval()

    with open(args.file, "rb") as f:
        meta_bytes = f.readline()
        encoded_vector = f.read()

    meta = meta_bytes.decode().split(":")
    max_value = float(meta[0])
    img_size = tuple(map(int, meta[1:]))

    bit_string = utils.bytes_to_bitstring(encoded_vector)[1:]

    coder = arithmetic_compressor.AECompressor(
        arithmetic_compressor.models.StaticModel(utils.get_uniform_statistics(args.b))
    )
    bit_vector = list(map(int, bit_string))
    decoded_vector = coder.decompress(bit_vector, utils.HIDDEN_SIZE)
    dequantized_vector = utils.dequantize(decoded_vector, args.b)
    from_0_1 = list(map(lambda x: x * max_value, dequantized_vector))

    image_np = (
        model(torch.FloatTensor([from_0_1]))[0]
        .moveaxis(0, 2)
        .detach()
        .cpu()
        .numpy()
    )

    image = Image.fromarray((255 * image_np).astype(np.uint8)).resize(img_size)
    image.save(args.save_to, format="PNG")


if __name__ == "__main__":
    args = _configure_arg_parser().parse_args()
    main(args)
