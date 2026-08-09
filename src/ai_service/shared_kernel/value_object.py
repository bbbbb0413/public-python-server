from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class ValueObject(ABC, Generic[T]):
    def __init__(self, value: T) -> None:
        self._validate(value)
        self._value = value

    @abstractmethod
    def _validate(self, value: T) -> None: ...

    def get_value(self) -> T:
        return self._value

    def equals(self, other: "ValueObject[T]") -> bool:
        return isinstance(other, self.__class__) and self._value == other._value
