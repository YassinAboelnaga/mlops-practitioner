from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PRODML_")

    data_path: str = "data/green_tripdata_2025-01.parquet"
    model_path: str = "models/model.pkl"
    report_path: str = "reports/module-2.md"
    test_size: float = 0.2
    random_state: int = 42
    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
