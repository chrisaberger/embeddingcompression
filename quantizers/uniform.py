import numpy as np
from .quantizer import Quantizer 

def _quantize(data, num_bits, scale_factor, biased=False):
    if not biased:
        random_data = np.random.uniform(0, 1, size=data.shape)
        data = np.floor((data / float(scale_factor)) + random_data)
    else:
        data = np.floor(data / float(scale_factor) + 0.5)
    min_value = -1 * (2**(num_bits - 1))
    max_value = 2**(num_bits - 1) - 1
    data = np.clip(data, min_value, max_value)
    return data * scale_factor

class UniformQuantizer(Quantizer):
    def __init__(self, num_bits):
        self.num_bits = num_bits

    def name(self):
        return "uniform" + str(self.num_bits) + "b"

    def quantize(self, X):
        num_bits = self.num_bits
        # Determine the center.
        min_val = np.amin(X)
        max_val = np.amax(X)

        center = (max_val - min_val) / 2
        center = max_val - center

        # Center around 0.
        X_recentered = X - center
        min_val = min_val - center
        max_val = max_val - center

        # Max and min values allowed with this number of bits.
        min_bit_value = -1 * (2**(num_bits - 1))
        max_bit_value = 2**(num_bits - 1) - 1

        # Determine scale factors needed to capture range.
        if max_bit_value == 0:
            sf_max = max_val
        else:
            sf_max = max_val / max_bit_value
        sf_min = min_val / min_bit_value

        # Select larger of the two scale factors.
        sf = max(sf_min, sf_max)

        # Actually quantize.
        compressed_X = _quantize(X, num_bits, sf)
        total_bytes = (compressed_X.size * num_bits) / 8

        # Recenter and return.
        compressed_X += center

        return compressed_X, total_bytes
