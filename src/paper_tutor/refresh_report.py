"""The Telegram text for a refresh result or failure, shared by the API (/refresh/status) and the CLI (refresh_job)."""

import os

# Values of these variables are replaced by *** in everything that is logged or sent
SECRET_VARS = ["TELEGRAM_BOT_TOKEN", "OPENALEX_API_KEY", "POSTGRES_PASSWORD", "GROQ_API_KEY", "API_KEY",
               "LANGSMITH_API_KEY"]


def redact(text):
    """Replace the values of the secret environment variables in text (an error can carry a URL with a key)."""
    for name in SECRET_VARS:
        value = os.getenv(name)
        if value and len(value) >= 4:
            text = text.replace(value, "***")
    return text


def message(result):
    """The Telegram text for a refresh result: one line with the counts, then one line per paper."""
    lines = [f"Weekly refresh: {result['added']} new papers ({result['candidates']} candidates)"]
    lines += [f"• {paper['title']} ({paper['score']})" for paper in result["papers"]]
    text = "\n".join(lines)
    return f"[dry run] {text}" if result["dry_run"] else text


def describe(error):
    """"<error type>: <message>", without the stack trace or secrets."""
    detail = redact(str(error)).strip()
    return type(error).__name__ + (f": {detail}" if detail else "")


def failure_message(error):
    return f"[refresh failed] {describe(error)}"
