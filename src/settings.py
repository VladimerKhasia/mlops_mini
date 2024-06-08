from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    HF_TOKEN: str
    WB_TOKEN: str


settings = Settings()