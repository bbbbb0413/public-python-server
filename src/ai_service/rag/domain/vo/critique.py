from dataclasses import dataclass, field

from ai_service.shared_kernel.value_object import ValueObject


@dataclass(frozen=True)
class CritiqueProps:
    answered: bool
    missing: list[str] = field(default_factory=list)
    next_query: str = ""
    confidence: float = 0.0


class Critique(ValueObject[CritiqueProps]):
    def _validate(self, value: CritiqueProps) -> None:
        if value.confidence < 0 or value.confidence > 1:
            raise ValueError("confidence는 0 이상 1 이하여야 합니다.")

    def is_satisfied(self, threshold: float) -> bool:
        return self.get_value().answered and self.get_value().confidence >= threshold

    def get_next_query(self) -> str:
        return self.get_value().next_query

    def get_confidence(self) -> float:
        return self.get_value().confidence

    def get_missing(self) -> list[str]:
        return list(self.get_value().missing)

    @classmethod
    def of(
        cls, answered: bool, missing: list[str], next_query: str, confidence: float
    ) -> "Critique":
        return cls(CritiqueProps(answered, missing, next_query, confidence))
