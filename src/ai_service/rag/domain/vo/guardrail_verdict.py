from dataclasses import dataclass

from ai_service.shared_kernel.value_object import ValueObject


@dataclass(frozen=True)
class VerdictValue:
    allowed: bool
    reason: str
    matched_pattern: str | None = None


class GuardrailVerdict(ValueObject[VerdictValue]):
    def _validate(self, value: VerdictValue) -> None:
        if not value.allowed and not value.reason:
            raise ValueError("차단 판정에는 사유가 필요합니다.")

    @classmethod
    def allow(cls) -> "GuardrailVerdict":
        return cls(VerdictValue(allowed=True, reason="ok"))

    @classmethod
    def block(cls, reason: str, pattern: str | None = None) -> "GuardrailVerdict":
        return cls(VerdictValue(allowed=False, reason=reason, matched_pattern=pattern))

    def is_allowed(self) -> bool:
        return self.get_value().allowed

    def get_reason(self) -> str:
        return self.get_value().reason
