import os


REQUIRED_DB_ENV = {
    "dbname": "SAMVISION_DB_NAME",
    "user": "SAMVISION_DB_USER",
    "password": "SAMVISION_DB_PASSWORD",
    "host": "SAMVISION_DB_HOST",
    "port": "SAMVISION_DB_PORT",
}


def get_db_config():
    missing = [
        env_name for env_name in REQUIRED_DB_ENV.values() if not os.getenv(env_name)
    ]
    if missing:
        raise RuntimeError(
            "Missing required database environment variables: " + ", ".join(missing)
        )

    return {key: os.environ[env_name] for key, env_name in REQUIRED_DB_ENV.items()}

