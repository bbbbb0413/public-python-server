from ai_service.shared_kernel.value_object import ValueObject


class SimilarityThreshold(ValueObject[float]):
    def _validate(self, value: float) -> None:
        if value < 0 or value > 1:
            raise ValueError("유사도 임계값은 0과 1 사이여야 합니다.")

    @classmethod
    def of(cls, value: float) -> "SimilarityThreshold":
        return cls(value)
