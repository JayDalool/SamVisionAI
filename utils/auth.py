from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass
from typing import Mapping, MutableMapping


HASH_ALGORITHM = "pbkdf2_sha256"
DEFAULT_HASH_ITERATIONS = 600_000
MIN_HASH_ITERATIONS = 100_000
MIN_SESSION_SECRET_LENGTH = 32

AUTH_USER_KEY = "samvision_auth_user"
AUTH_TOKEN_KEY = "samvision_auth_token"


@dataclass(frozen=True)
class AuthConfig:
    username: str
    password_hash: str
    session_secret: str


def load_auth_config(env: Mapping[str, str] | None = None) -> AuthConfig:
    source = env if env is not None else os.environ
    return AuthConfig(
        username=source.get("SAMVISION_ADMIN_USERNAME", "").strip(),
        password_hash=source.get("SAMVISION_ADMIN_PASSWORD_HASH", "").strip(),
        session_secret=source.get("SAMVISION_SESSION_SECRET", "").strip(),
    )


def hash_password(
    password: str,
    *,
    iterations: int = DEFAULT_HASH_ITERATIONS,
    salt: bytes | None = None,
) -> str:
    if not password:
        raise ValueError("Password must not be empty.")
    if iterations < MIN_HASH_ITERATIONS:
        raise ValueError(f"Use at least {MIN_HASH_ITERATIONS} iterations.")

    salt = salt if salt is not None else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return f"{HASH_ALGORITHM}${iterations}${salt.hex()}${digest.hex()}"


def _parse_password_hash(password_hash: str) -> tuple[int, bytes, bytes] | None:
    parts = password_hash.split("$")
    if len(parts) != 4 or parts[0] != HASH_ALGORITHM:
        return None

    try:
        iterations = int(parts[1])
        salt = bytes.fromhex(parts[2])
        expected_digest = bytes.fromhex(parts[3])
    except ValueError:
        return None

    if iterations < MIN_HASH_ITERATIONS or len(salt) < 16 or len(expected_digest) != 32:
        return None
    return iterations, salt, expected_digest


def is_supported_password_hash(password_hash: str) -> bool:
    return _parse_password_hash(password_hash) is not None


def verify_password(password: str, password_hash: str) -> bool:
    parsed = _parse_password_hash(password_hash)
    if parsed is None:
        return False

    iterations, salt, expected_digest = parsed
    candidate_digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(candidate_digest, expected_digest)


def auth_config_errors(config: AuthConfig) -> list[str]:
    errors: list[str] = []
    if not config.username:
        errors.append("SAMVISION_ADMIN_USERNAME is missing.")
    if not config.password_hash:
        errors.append("SAMVISION_ADMIN_PASSWORD_HASH is missing.")
    elif not is_supported_password_hash(config.password_hash):
        errors.append("SAMVISION_ADMIN_PASSWORD_HASH is not a supported PBKDF2 hash.")
    if not config.session_secret:
        errors.append("SAMVISION_SESSION_SECRET is missing.")
    elif len(config.session_secret) < MIN_SESSION_SECRET_LENGTH:
        errors.append(
            f"SAMVISION_SESSION_SECRET must be at least {MIN_SESSION_SECRET_LENGTH} characters."
        )
    return errors


def validate_credentials(username: str, password: str, config: AuthConfig) -> bool:
    if auth_config_errors(config):
        return False
    if not hmac.compare_digest(username.strip(), config.username):
        return False
    return verify_password(password, config.password_hash)


def issue_session_token(
    username: str,
    session_secret: str,
    *,
    issued_at: int | None = None,
    nonce: str | None = None,
) -> str:
    issued_at = issued_at if issued_at is not None else int(time.time())
    nonce = nonce if nonce is not None else secrets.token_urlsafe(24)
    signature = _session_signature(username, issued_at, nonce, session_secret)
    return f"{issued_at}:{nonce}:{signature}"


def _session_signature(
    username: str,
    issued_at: int,
    nonce: str,
    session_secret: str,
) -> str:
    payload = f"{username}|{issued_at}|{nonce}".encode("utf-8")
    return hmac.new(
        session_secret.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()


def validate_session_token(token: str, username: str, session_secret: str) -> bool:
    try:
        issued_at_raw, nonce, signature = token.split(":", 2)
        issued_at = int(issued_at_raw)
    except (AttributeError, TypeError, ValueError):
        return False

    expected_signature = _session_signature(username, issued_at, nonce, session_secret)
    return hmac.compare_digest(signature, expected_signature)


def mark_authenticated(
    session_state: MutableMapping[str, object],
    username: str,
    session_secret: str,
) -> None:
    session_state[AUTH_USER_KEY] = username
    session_state[AUTH_TOKEN_KEY] = issue_session_token(username, session_secret)


def clear_authentication(session_state: MutableMapping[str, object]) -> None:
    session_state.pop(AUTH_USER_KEY, None)
    session_state.pop(AUTH_TOKEN_KEY, None)


def is_authenticated(
    session_state: Mapping[str, object],
    config: AuthConfig,
) -> bool:
    if auth_config_errors(config):
        return False

    username = session_state.get(AUTH_USER_KEY)
    token = session_state.get(AUTH_TOKEN_KEY)
    if not isinstance(username, str) or not isinstance(token, str):
        return False
    if not hmac.compare_digest(username, config.username):
        return False
    return validate_session_token(token, config.username, config.session_secret)
