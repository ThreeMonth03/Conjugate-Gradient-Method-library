"""Check rectangular products through the installed legacy Matrix API."""

import numpy as np
from _cgpy.Matrix import Accelerated_Matrix, Naive_Matrix


def main():
    values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    vector = np.array([2.0, -1.0, 3.0])
    for matrix_type in (Naive_Matrix, Accelerated_Matrix):
        matrix = matrix_type(values)
        product = matrix @ matrix_type(vector)
        np.testing.assert_allclose(product.tolist(), values @ vector)
        print(f"{matrix!r} @ vector = {product.tolist()}")


if __name__ == "__main__":
    main()
