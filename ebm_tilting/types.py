from typing import Callable

from jaxtyping import Float32
from torch import Tensor

Scalar = Float32[Tensor, ""]
BatchedScalar = Float32[Tensor, "B"]
MaybeBatchedScalar = BatchedScalar | Scalar
ExpandedScalar = Float32[Tensor, "B 1 1 1"]
BatchedScalarFun = Callable[[BatchedScalar], BatchedScalar]
Image = Float32[Tensor, "1 W H"]
BatchedImage = Float32[Tensor, "B 1 W H"]
BatchedIntegratorSolution = Tensor
