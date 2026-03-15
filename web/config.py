import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # SECRET_KEY: must be set via environment variable in production
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-change-in-production")

    # DATABASE_URL is injected by docker-compose as:
    #   mysql+mysqlconnector://user:pass@db:3306/autotos
    # Falls back to localhost for running outside Docker (local dev)
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        "mysql+mysqlconnector://root:@localhost:3306/autotos"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # AI service URL — docker-compose sets this to http://ai:8000
    # Falls back to localhost for running outside Docker
    AUTO_TOS_MODEL_URL     = os.environ.get("AUTO_TOS_MODEL_URL", "http://localhost:8000")
    AUTO_TOS_MODEL_TIMEOUT = int(os.environ.get("AUTO_TOS_MODEL_TIMEOUT", "120"))