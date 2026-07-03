from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_GEMINI_MODELS = (
    "gemini-3.5-flash,"
    "gemini-3.0-flash,"
    "gemini-2.5-flash,"
    "gemini-2.5-flash-lite,"
    "gemini-3.1-flash-lite"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "AI Data Analyst"
    debug: bool = False
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    data_dir: Path = Path("data")
    upload_dir: Path = Path("data/uploads")
    processed_dir: Path = Path("data/processed")

    # Google Gemini — comma-separated keys for rotation on rate limits
    gemini_api_keys: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_models: str = DEFAULT_GEMINI_MODELS
    gemini_max_retries: int = 2

    max_upload_size_mb: int = 50
    preview_rows: int = 20

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def gemini_api_key_list(self) -> list[str]:
        return [key.strip() for key in self.gemini_api_keys.split(",") if key.strip()]

    @property
    def gemini_model_list(self) -> list[str]:
        models: list[str] = []
        for model in self.gemini_models.split(","):
            name = model.strip()
            if name and name not in models:
                models.append(name)
        return models


settings = Settings()
