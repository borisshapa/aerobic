import math
import os

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

TRANSFORM = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

HIDDEN_SIZE = 25088


def get_uniform_statistics(b: int) -> dict[int, float]:
    total = (1 << b) + 1
    return {symbol: 1 / total for symbol in range(0, total + 1)}


def get_image_tensor(image_path: str) -> torch.Tensor:
    img = Image.open(image_path)
    return transforms.ToTensor()(img)


def psnr(original_path: str, decompressed_path: str) -> float:
    original_t = get_image_tensor(original_path)
    decompressed_t = get_image_tensor(decompressed_path)
    psnr_t = 20 * torch.log10(255.0 / torch.nn.MSELoss()(original_t, decompressed_t))
    return psnr_t.item()


def bpp(original_path: str, compressed_path: str):
    original_size = Image.open(original_path).size
    pixels_count = original_size[0] * original_size[1]
    return os.path.getsize(compressed_path) * 8 / pixels_count


def quantize_float(value: float, b: int) -> int:
    return math.floor(value * (1 << b) + 0.5)


def dequantize_int(value: int, b: int) -> float:
    return value / (1 << b)


def quantize(vector: list[float], b: int) -> list[int]:
    return [quantize_float(val, b) for val in vector]


def dequantize(vector: list[int], b: int) -> list[float]:
    return [dequantize_int(val, b) for val in vector]


def bitstring_to_bytes(s: str) -> bytes:
    return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder="big")


def bytes_to_bitstring(b: bytes) -> str:
    return format(int.from_bytes(b, "big"), "b")


def to_0_1(x: list[float]) -> tuple[list[float], float]:
    a = np.array(x)
    p = np.percentile(a, 99)

    def value_to_0_1(v: float):
        if v > p:
            v = p
        return v / p

    return [value_to_0_1(v) for v in x], p


def from_0_1(x: float) -> float:
    return math.log(x / (1 - x))
