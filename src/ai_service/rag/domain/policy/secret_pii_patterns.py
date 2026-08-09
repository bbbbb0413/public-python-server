import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SecretPiiPattern:
    label: str
    pattern: re.Pattern[str]


# 응답 유출(exfiltration) 방지를 위한 시크릿/PII 탐지 패턴 (OWASP LLM02:2025)
SECRET_PII_PATTERNS: list[SecretPiiPattern] = [
    SecretPiiPattern("OPENAI_API_KEY", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    SecretPiiPattern("AWS_ACCESS_KEY", re.compile(r"AKIA[0-9A-Z]{16}")),
    SecretPiiPattern("JWT", re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")),
    SecretPiiPattern("BEARER_TOKEN", re.compile(r"Bearer\s+[A-Za-z0-9._-]{10,}", re.IGNORECASE)),
    SecretPiiPattern(
        "PRIVATE_KEY",
        re.compile(r"-----BEGIN[ A-Z]*PRIVATE KEY-----[\s\S]*?-----END[ A-Z]*PRIVATE KEY-----"),
    ),
    SecretPiiPattern("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    SecretPiiPattern("KR_RRN", re.compile(r"\b\d{6}-?[1-4]\d{6}\b")),
    SecretPiiPattern("KR_PHONE", re.compile(r"01[016789]-?\d{3,4}-?\d{4}")),
]
