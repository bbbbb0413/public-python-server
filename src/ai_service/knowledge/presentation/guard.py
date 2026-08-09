from typing import Annotated

from fastapi import Depends, Header, HTTPException

from ai_service.config.settings import Settings, get_settings


async def require_admin_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_admin_key: Annotated[str | None, Header()] = None,
) -> None:
    if not settings.admin_api_key:
        return
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid admin API key")
