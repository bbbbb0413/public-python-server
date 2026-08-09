import uuid

from ai_service.shared_kernel.value_object import ValueObject


class SessionId(ValueObject[str]):
    def _validate(self, value: str) -> None:
        if not value or not value.strip():
            raise ValueError("SessionId는 비어있을 수 없습니다.")

    @classmethod
    def generate(cls) -> "SessionId":
        return cls(str(uuid.uuid4()))

    @classmethod
    def of(cls, value: str) -> "SessionId":
        return cls(value)
