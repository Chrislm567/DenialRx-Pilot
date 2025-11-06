from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "DenialRx Appeals Service"
    environment: str = "local"
    audit_log_path: str = "data/audit.log"
    evidence_pack_dir: str = "data/evidence"

    model_config = SettingsConfigDict(env_prefix="APPEALS_", env_file=".env")


settings = Settings()
