
__all__ = [
    "prod_settings",
    "dev_settings",
    "log_config", 
    ]

from pydantic import BaseSettings
from logging import (
    DEBUG as _debug,
    INFO as _info,
    WARNING as _warning,
    ERROR as _err,
    CRITICAL as _critical,
    )


class CommonSettings(BaseSettings):
    APP_NAME: str = "API Demo"
    APP_ENTRYPOINT: str = "src.main:app"
    DEBUG: bool = False
    LOG_LEVEL: int = None # 'critical': 50, 'error': 40, 'warning': 30, 'info': 20, 'debug': 10
    RELOAD_ON_SAVE: bool = False
    PORT: int = 8000


class DevelopmentMixin(BaseSettings):
    DEBUG: bool = True
    HOST: str = "127.0.0.1"
    LOG_LEVEL: int = _debug
    RELOAD_ON_SAVE: bool = True


class ProductionMixin(BaseSettings):
    HOST: str = "0.0.0.0"
    LOG_LEVEL: int = _err


class ProductionSettings(ProductionMixin, CommonSettings):
    pass


class DevelopmentSettings(DevelopmentMixin, CommonSettings):
    pass


prod_settings = ProductionSettings()
dev_settings = DevelopmentSettings()