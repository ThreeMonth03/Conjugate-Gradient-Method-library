"""Validate solver-specific shape contracts before convenience allocation."""


def validate_operator_shape(shape: tuple[int, ...]) -> None:
    if len(shape) != 2 or shape[0] == 0 or shape[0] != shape[1]:
        raise ValueError("A must be a nonempty square matrix")
