from pydantic_settings import BaseSettings, SettingsConfigDict

# clasa folosita pentru a gestiona si incarca setarile
# noi incarcam din .env app_secret_key
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    app_secret_key: str = 'APP_SECRET_KEY'
