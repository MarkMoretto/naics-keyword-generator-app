
__all__ = [
    "Dict",
    "List",
    "Number",
    "Vector",
    "MutVector",
    "Iterables",
    "StrDict",
    "Tuple",
]


from typing import Dict, Iterable, List, Optional, Union, Tuple

# Number for either integer or foating point value.
Number = Union[int, float]

# Numerical vector (mutable)
Vector = List[Number]
MutVector = Tuple[Number]

Iterables = Union[Iterable, Iterable]
StrDict = Dict[str, str]
