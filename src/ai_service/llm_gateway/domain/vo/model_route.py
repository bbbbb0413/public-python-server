from ai_service.shared_kernel.value_object import ValueObject


class ModelRoute(ValueObject[str]):
    def _validate(self, value: str) -> None:
        if not value or not value.strip():
            raise ValueError("모델명은 빈 값일 수 없습니다.")

    @classmethod
    def of(cls, model: str) -> "ModelRoute":
        return cls(model)
