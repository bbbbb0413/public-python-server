from ai_service.rag.domain.policy.secret_pii_patterns import SECRET_PII_PATTERNS


class SecretPiiScanner:
    def mask(self, text: str) -> str:
        masked = text
        for entry in SECRET_PII_PATTERNS:
            masked = entry.pattern.sub(f"[REDACTED_{entry.label}]", masked)
        return masked

    def contains_sensitive_data(self, text: str) -> bool:
        return any(entry.pattern.search(text) for entry in SECRET_PII_PATTERNS)
