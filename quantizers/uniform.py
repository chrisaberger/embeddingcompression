import numpy as np
from .quantizer import Quantizer


class UniformQuantizer(Quantizer):

    def __init__(self, num_bits):
        self.num_bits = num_bits

    def get_total_bytes(self, X):
        total_bytes = (X.size * self.num_bits) / 8
        total_bytes += (32 / 8) * 2    #store scale factor and center
        return total_bytes


class FixedPointQuantizer(UniformQuantizer):

    def name(self):
        return "uniform_fp" + str(self.num_bits) + "b"

    def _quantize(self, data, num_bits, scale_factor, biased=False):
        if not biased:
            random_data = np.random.uniform(0, 1, size=data.shape)
            data = np.floor((data / float(scale_factor)) + random_data)
        else:
            data = np.floor(data / float(scale_factor) + 0.5)
        min_value = -1 * (2**(num_bits - 1))
        max_value = 2**(num_bits - 1) - 1
        data = np.clip(data, min_value, max_value)
        return data * scale_factor

    def quantize(self, X):
        total_bytes = UniformQuantizer.get_total_bytes(self, X)

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
        min_bit_value = -1 * (2**(self.num_bits - 1))
        max_bit_value = 2**(self.num_bits - 1) - 1

        # Determine scale factors needed to capture range.
        if max_bit_value == 0:
            sf_max = max_val
        else:
            sf_max = max_val / max_bit_value
        if min_bit_value == 0:
            sf_min = min_val
        else:
            sf_min = min_val / min_bit_value

        # Select larger of the two scale factors.
        sf = max(sf_min, sf_max)

        if sf == 0.0:
            return np.zeros(X.shape) + center, total_bytes

        # Actually quantize.
        compressed_X = self._quantize(X_recentered, self.num_bits, sf)

        # Recenter and return.
        compressed_X += center

        return compressed_X, total_bytes


class MidtreadQuantizer(UniformQuantizer):

    def name(self):
        return "uniform_mt" + str(self.num_bits) + "b"

    def quantize(self, X):
        if (self.num_bits <= 1):    #mid-tread requires at least 2 bits
            return np.zeros(X.shape), UniformQuantizer.get_total_bytes(self, X)

        L = 2**self.num_bits - 1
        eps = 1e-7
        a = max(np.max(X), -1 * np.min(X)) + eps
        delta = 2 * a / (L - 1)
        return np.round(
            (X + a) / delta) * delta - a, UniformQuantizer.get_total_bytes(
                self, X)
