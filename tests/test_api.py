import threading
import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from paper_tutor import api
from paper_tutor.api import require_api_key
from paper_tutor.refresh_report import message


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



RESULT = {"added": 1, "dry_run": False, "candidates": 1187, "already_known": 0,
          "papers": [{"id": "W1", "title": "Lasso paths", "year": 2026, "score": 0.822, "topic_id": "T1"}]}


@pytest.fixture
def refresh_api(monkeypatch):
    """A test client with a fake refresh that runs until release is set; returns (client, release, calls)."""
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setattr(api, "state", {"config": {}, "embed_model": None, "model_cfg": {}})
    monkeypatch.setattr(api, "refresh_lock", threading.Lock())  # a fresh lock, so no test blocks another
    release, calls = threading.Event(), []

    def fake_refresh(config, model, model_cfg, dry_run):
        calls.append(dry_run)
        assert release.wait(5), "the test never released the refresh"
        return {**RESULT, "dry_run": dry_run, "added": 0 if dry_run else 1}

    monkeypatch.setattr(api, "refresh_papers", fake_refresh)
    yield TestClient(api.app), release, calls  # without "with", so the real models are not loaded
    release.set()  # never leave a thread waiting


def wait_until_finished(client):
    for _ in range(100):
        status = client.get("/refresh/status").json()
        if status["state"] != "running":
            return status
        time.sleep(0.02)
    raise AssertionError("the refresh did not finish")


def test_status_is_404_before_any_refresh(refresh_api):
    client, _, _ = refresh_api
    assert client.get("/refresh/status").status_code == 404


def test_refresh_starts_in_the_background_and_reports_the_result(refresh_api):
    client, release, calls = refresh_api
    response = client.post("/refresh")
    assert response.status_code == 202
    run = response.json()
    assert run["state"] == "running" and run["dry_run"] is False
    assert client.get("/refresh/status").json()["run_id"] == run["run_id"]  # still running: answered at once

    release.set()
    status = wait_until_finished(client)
    assert status["run_id"] == run["run_id"]
    assert status["state"] == "done"
    assert status["result"] == RESULT
    assert status["message"] == message(RESULT)
    assert calls == [False]


def test_dry_run_is_passed_on(refresh_api):
    client, release, calls = refresh_api
    client.post("/refresh?dry_run=true")
    release.set()
    assert wait_until_finished(client)["message"].startswith("[dry run] Weekly refresh: 0 new papers")
    assert calls == [True]


def test_second_refresh_while_one_runs_gets_409(refresh_api):
    client, release, calls = refresh_api
    first = client.post("/refresh").json()
    second = client.post("/refresh")
    assert second.status_code == 409
    assert second.json()["run_id"] == first["run_id"]
    release.set()
    wait_until_finished(client)
    assert calls == [False]  # the second request started nothing
    assert client.post("/refresh").status_code == 202  # the lock is free again


def test_failed_refresh_reports_the_error_without_secrets(refresh_api, monkeypatch):
    client, _, _ = refresh_api
    monkeypatch.setenv("OPENALEX_API_KEY", "openalex-key-1234")

    def failing_refresh(config, model, model_cfg, dry_run):
        raise RuntimeError("GET https://api.openalex.org/works?api_key=openalex-key-1234 failed")

    monkeypatch.setattr(api, "refresh_papers", failing_refresh)
    client.post("/refresh")
    status = wait_until_finished(client)
    assert status["state"] == "failed"
    assert status["message"] == "[refresh failed] RuntimeError: GET https://api.openalex.org/works?api_key=*** failed"
    assert "openalex-key-1234" not in str(status)
    assert client.post("/refresh").status_code == 202  # the lock is free again after a failure


def test_refresh_endpoints_need_the_key(refresh_api, monkeypatch):
    client, _, _ = refresh_api
    monkeypatch.setenv("API_KEY", "secret")
    assert client.post("/refresh").status_code == 401
    assert client.get("/refresh/status").status_code == 401
