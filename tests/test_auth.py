from __future__ import annotations

import os
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from utils.auth import (
    AuthConfig,
    auth_config_errors,
    clear_authentication,
    hash_password,
    is_authenticated,
    mark_authenticated,
    validate_credentials,
    verify_password,
)
from utils.db_config import get_db_config


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_PATH = ROOT_DIR / "app" / "streamlit_app.py"
TEST_PASSWORD = "correct horse battery staple"
TEST_HASH = hash_password(
    TEST_PASSWORD,
    iterations=100_000,
    salt=bytes.fromhex("00112233445566778899aabbccddeeff"),
)
TEST_SECRET = "test-session-secret-that-is-long-enough"


def auth_env() -> dict[str, str]:
    return {
        "SAMVISION_ADMIN_USERNAME": "admin",
        "SAMVISION_ADMIN_PASSWORD_HASH": TEST_HASH,
        "SAMVISION_SESSION_SECRET": TEST_SECRET,
        "SAMVISION_DB_NAME": "SamVision",
        "SAMVISION_DB_USER": "postgres",
        "SAMVISION_DB_PASSWORD": "test-only",
        "SAMVISION_DB_HOST": "127.0.0.1",
        "SAMVISION_DB_PORT": "5432",
    }


@contextmanager
def streamlit_app_test_context():
    rental_stub = types.ModuleType("utils.rental_income_tool")
    rental_stub.render_rental_income_tab = lambda: None
    with patch.dict(os.environ, auth_env(), clear=False), patch.dict(
        sys.modules,
        {"utils.rental_income_tool": rental_stub},
    ):
        yield


class AuthHelperTests(unittest.TestCase):
    def test_password_hash_verifies_correct_password_only(self) -> None:
        self.assertTrue(verify_password(TEST_PASSWORD, TEST_HASH))
        self.assertFalse(verify_password("wrong password", TEST_HASH))

    def test_config_requires_all_auth_variables(self) -> None:
        config = AuthConfig(username="", password_hash="", session_secret="")
        errors = auth_config_errors(config)
        self.assertEqual(len(errors), 3)

    def test_credentials_and_session_state(self) -> None:
        config = AuthConfig(
            username="admin",
            password_hash=TEST_HASH,
            session_secret=TEST_SECRET,
        )
        self.assertTrue(validate_credentials("admin", TEST_PASSWORD, config))
        self.assertFalse(validate_credentials("admin", "wrong password", config))

        state: dict[str, object] = {}
        self.assertFalse(is_authenticated(state, config))
        mark_authenticated(state, "admin", TEST_SECRET)
        self.assertTrue(is_authenticated(state, config))
        clear_authentication(state)
        self.assertFalse(is_authenticated(state, config))


class DatabaseConfigTests(unittest.TestCase):
    def test_db_config_uses_required_samvision_env_vars(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SAMVISION_DB_NAME": "SamVision",
                "SAMVISION_DB_USER": "samvision",
                "SAMVISION_DB_PASSWORD": "secret",
                "SAMVISION_DB_HOST": "samvision-postgres",
                "SAMVISION_DB_PORT": "5432",
                "POSTGRES_PASSWORD": "ignored",
            },
            clear=True,
        ):
            self.assertEqual(
                get_db_config(),
                {
                    "dbname": "SamVision",
                    "user": "samvision",
                    "password": "secret",
                    "host": "samvision-postgres",
                    "port": "5432",
                },
            )

    def test_db_config_requires_all_samvision_env_vars(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "SAMVISION_DB_PASSWORD"):
                get_db_config()


class StreamlitAuthGateTests(unittest.TestCase):
    def test_unauthenticated_user_sees_login_only(self) -> None:
        with streamlit_app_test_context():
            app = AppTest.from_file(str(APP_PATH), default_timeout=15)
            app.run()

        self.assertEqual(app.title[0].value, "SamVision AI")
        self.assertEqual(app.subheader[0].value, "Sign in")
        self.assertEqual(len(app.tabs), 0)

    def test_wrong_login_fails(self) -> None:
        with streamlit_app_test_context():
            app = AppTest.from_file(str(APP_PATH), default_timeout=15)
            app.run()
            app.text_input[0].input("admin")
            app.text_input[1].input("wrong password")
            app.button[0].click()
            app.run()

        self.assertTrue(any("Invalid username or password." in item.value for item in app.error))
        self.assertEqual(len(app.tabs), 0)

    def test_correct_login_and_logout_work(self) -> None:
        with streamlit_app_test_context():
            app = AppTest.from_file(str(APP_PATH), default_timeout=30)
            app.run()
            app.text_input[0].input("admin")
            app.text_input[1].input(TEST_PASSWORD)
            app.button[0].click()
            app.run()

            self.assertTrue(any("SamVision AI" in item.value for item in app.title))
            self.assertGreater(len(app.tabs), 0)

            app.sidebar.button[0].click()
            app.run()

        self.assertEqual(app.title[0].value, "SamVision AI")
        self.assertEqual(app.subheader[0].value, "Sign in")
        self.assertEqual(len(app.tabs), 0)


if __name__ == "__main__":
    unittest.main()
