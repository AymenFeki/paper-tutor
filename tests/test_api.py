import pytest
from fastapi import HTTPException

from paper_tutor.api import require_api_key


def test_no_key_configured_allows_everything(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    require_api_key(None)


def test_wrong_or_missing_key_is_rejected(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    for key in [None, "wrong"]:
        with pytest.raises(HTTPException):
            require_api_key(key)


def test_right_key_passes(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    require_api_key("secret")