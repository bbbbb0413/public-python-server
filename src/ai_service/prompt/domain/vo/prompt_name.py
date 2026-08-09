import re

from ai_service.shared_kernel.value_object import ValueObject

_NAME_PATTERN = re.compile(r"^[a-z0-9-]+$")


class PromptName(ValueObject[str]):
    def _validate(self, value: str) -> None:
        if not value or not value.strip():
            raise ValueError("프롬프트 이름은 비어있을 수 없습니다.")
        if not _NAME_PATTERN.match(value):
            raise ValueError("프롬프트 이름은 소문자, 숫자, 하이픈만 허용됩니다.")

    @classmethod
    def of(cls, value: str) -> "PromptName":
        return cls(value)
