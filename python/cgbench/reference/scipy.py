"""Public SciPy CSR matvec with NumPy vector operations."""

from . import numpy


class ScipyOps(numpy.NumpyOps):
    """Public CSR matvec with preallocated NumPy vector updates.

    SciPy's public ``A @ x`` allocates a product. Keep that cost visible rather
    than depending on private sparsetools or claiming an allocation-free solve.
    """

    def __init__(self, arrays):
        super().__init__(arrays)
        if not hasattr(self.matrix, "indptr") or self.matrix.format != "csr":
            raise ValueError("scipy-out requires canonical CSR inputs")
        if not self.matrix.has_canonical_format or not self.np.all(
            self.np.isfinite(self.matrix.data)
        ):
            raise ValueError("CSR inputs must be canonical and finite")

    def validate(self, api):
        # All implementations receive a prepared, validated CSR object. Input
        # conversion and canonical validation belong outside both solve scopes.
        super().validate(False)

    def matvec(self, vector, out):
        self.np.copyto(out, self.matrix @ vector)
