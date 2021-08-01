
# __all__ = [
#     "Dict",
#     "List",
#     "Number",
#     "Vector",
#     "MutVector",
#     "Iterables",
#     "StrDict",
#     "StrList",
#     "Tuple",
# ]


from typing import Dict, Iterable, Iterator, List, Optional, Union, Tuple

# Number for either integer or foating point value.
Number = Union[int, float]

Vector = List[Number]
NumList = List[Number]
IntList = List[int]
FloatList = List[float]
StrList = List[str]

MutVector = Tuple[Number]

Iterables = Union[Iterable, Iterable]


StrDict = Dict[str, str]
StrNumDict = Dict[str, Number]
NumDict = Dict[Number, Number]
IntDict = Dict[int, int]
