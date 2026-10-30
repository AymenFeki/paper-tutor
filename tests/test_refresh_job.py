"""Tests for the refresh job: the Telegram messages and exit codes (refresh and httpx are mocked)."""

from unittest.mock import MagicMock

import httpx
import pytest

from paper_tutor import refresh_job
from paper_tutor.refresh_job import failure_message, main, message

PAPERS = [{"id": "W1", "title": "Lasso paths", "year": 2026, "score": 0.822, "topic_id": "T1"},
          {"id": "W2", "title": "Deep sets", "year": 2026, "score": 0.81, "topic_id": "T2"}]


def result(papers=(), dry_run=False):
    return {"added": 0 if dry_run else len(papers), "dry_run": dry_run, "papers": list(papers),
            "candidates": 1187, "already_known": 0}


@pytest.fixture
def job(monkeypatch):
    """Patch the config, model and refresh; returns the mocks for refresh and httpx.post."""
    monkeypatch.setattr(refresh_job, "load_config", lambda: {"embeddings": {}})
    monkeypatch.setattr(refresh_job, "active_model", lambda config: ("bge-small", {}))
    monkeypatch.setattr(refresh_job, "load_model", lambda model_cfg: None)
    refresh = MagicMock(return_value=result(PAPERS))
    monkeypatch.setattr(refresh_job, "refresh", refresh)
    post = MagicMock(return_value=httpx.Response(200, request=httpx.Request("POST", "https://api.telegram.org")))
    monkeypatch.setattr(refresh_job.httpx, "post", post)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:secret-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    return refresh, post


def sent_text(post):
    return post.call_args.kwargs["json"]["text"]


def test_message_without_papers():
    assert message(result()) == "Weekly refresh: 0 new papers (1187 candidates)"


def test_message_lists_every_paper_with_its_score():
    assert message(result(PAPERS)) == (
        "Weekly refresh: 2 new papers (1187 candidates)\n• Lasso paths (0.822)\n• Deep sets (0.81)"
    )


def test_message_of_a_dry_run_has_a_prefix():
    assert message(result(PAPERS, dry_run=True)).startswith("[dry run] Weekly refresh: 0 new papers (1187 candidates)\n")


def test_job_sends_the_result_and_exits_with_0(job):
    refresh, post = job
    assert main([]) == 0
    assert refresh.call_args.kwargs == {"dry_run": False}
    assert post.call_args.args[0] == "https://api.telegram.org/bot123:secret-token/sendMessage"
    assert post.call_args.kwargs["json"]["chat_id"] == "42"
    assert sent_text(post) == message(result(PAPERS))


def test_dry_run_is_passed_on(job):
    refresh, post = job
    refresh.return_value = result(PAPERS, dry_run=True)
    assert main(["--dry-run"]) == 0
    assert refresh.call_args.kwargs == {"dry_run": True}
    assert sent_text(post).startswith("[dry run] ")


@pytest.mark.parametrize("missing", ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"])
def test_without_token_or_chat_the_message_is_only_logged(job, monkeypatch, capsys, missing):
    _, post = job
    monkeypatch.delenv(missing)
    assert main([]) == 0
    post.assert_not_called()
    assert "• Lasso paths (0.822)" in capsys.readouterr().out


def test_failure_sends_an_alert_without_secrets_and_exits_with_1(job, monkeypatch):
    refresh, post = job
    monkeypatch.setenv("OPENALEX_API_KEY", "openalex-key-1234")
    refresh.side_effect = RuntimeError("GET https://api.openalex.org/works?api_key=openalex-key-1234 failed")
    assert main([]) == 1
    assert sent_text(post) == "[refresh failed] RuntimeError: GET https://api.openalex.org/works?api_key=*** failed"


def test_failure_message_without_detail():
    assert failure_message(KeyError()) == "[refresh failed] KeyError"


def test_failure_without_token_is_logged_and_exits_with_1(job, monkeypatch, capsys):
    refresh, post = job
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN")
    refresh.side_effect = ValueError("no candidates")
    assert main([]) == 1
    post.assert_not_called()
    assert "[refresh failed] ValueError: no candidates" in capsys.readouterr().out


def test_telegram_error_is_logged_without_the_token_and_exits_with_1(job, capsys):
    _, post = job
    post.side_effect = httpx.ConnectError("cannot reach https://api.telegram.org/bot123:secret-token/sendMessage")
    assert main([]) == 1
    out = capsys.readouterr().out
    assert "Telegram failed: ConnectError: cannot reach https://api.telegram.org/bot***/sendMessage" in out
    assert "secret-token" not in out


def test_telegram_status_error_exits_with_1(job):
    _, post = job
    post.return_value = httpx.Response(400, request=httpx.Request("POST", "https://api.telegram.org"))
    assert main([]) == 1


def test_cli_exits_with_the_code_of_main(job, monkeypatch):
    refresh, _ = job
    refresh.side_effect = RuntimeError("boom")
    monkeypatch.setattr(refresh_job.sys, "argv", ["paper-tutor-refresh", "--dry-run"])
    with pytest.raises(SystemExit) as exit_info:
        refresh_job.cli()
    assert exit_info.value.code == 1
    assert refresh.call_args.kwargs == {"dry_run": True}


@pytest.mark.parametrize("argv", [["--dryrun"], ["--help"]])
def test_unknown_arguments_and_help_never_start_a_refresh(job, argv):
    refresh, post = job
    with pytest.raises(SystemExit):
        main(argv)
    refresh.assert_not_called()
    post.assert_not_called()
