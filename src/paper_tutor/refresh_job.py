"""Weekly refresh as an Azure Container Apps job (#37): runs refresh() outside an HTTP request, so the
request timeout of Container Apps (about 4 minutes) no longer applies, and sends the result to Telegram.

    python -m paper_tutor.refresh_job             # add the papers and send the result
    python -m paper_tutor.refresh_job --dry-run   # only show what would be added ("[dry run]" message)

In the image the same command is installed as paper-tutor-refresh (see [project.scripts]): the Container Apps
job runs that, because az reads arguments that start with "-" (like "-m") as its own options.

Without TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID the message is only logged. Any error sends
"[refresh failed] <error type>: <message>" instead and exits with 1; so does a failed Telegram call.
"""

import argparse
import json
import os
import sys
import time
import traceback

import httpx

from paper_tutor.corpus import load_config
from paper_tutor.embed import active_model, load_model
from paper_tutor.refresh import refresh
from paper_tutor.tracing import git_sha

# Values of these variables are replaced by *** in everything the job logs or sends
SECRET_VARS = ["TELEGRAM_BOT_TOKEN", "OPENALEX_API_KEY", "POSTGRES_PASSWORD", "GROQ_API_KEY", "API_KEY",
               "LANGSMITH_API_KEY"]
TELEGRAM_LIMIT = 4096  # characters per message


def log(text):
    print(text, flush=True)  # flush, so the Container Apps log shows each line as it happens


def redact(text):
    """Replace the values of the secret environment variables in text (an error can carry a URL with a key)."""
    for name in SECRET_VARS:
        value = os.getenv(name)
        if value and len(value) >= 4:
            text = text.replace(value, "***")
    return text


def message(result):
    """The Telegram text for a refresh result, the same as .github/workflows/refresh.yml sends."""
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


def notify(text):
    """Send text to Telegram, or only log it if the bot token or chat id is not set. Returns False if sending failed."""
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        log(f"TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set, message not sent:\n{text}")
        return True
    try:
        response = httpx.post(f"https://api.telegram.org/bot{token}/sendMessage",
                              json={"chat_id": chat, "text": text[:TELEGRAM_LIMIT]}, timeout=30)
        response.raise_for_status()
    except httpx.HTTPError as error:
        log(f"Telegram failed: {describe(error)}")
        return False
    log("Sent to Telegram")
    return True


def parse_args(argv):
    """Unknown arguments are an error (exit code 2), so a typo like --dryrun never starts a real refresh."""
    parser = argparse.ArgumentParser(prog="paper-tutor-refresh", description="Weekly refresh, reported to Telegram.")
    parser.add_argument("--dry-run", action="store_true", help="only show what would be added, write nothing")
    return parser.parse_args(argv)


def main(argv):
    """Run the refresh and report it. Returns the exit code: 0 if it ran and the message was sent, else 1."""
    dry_run = parse_args(argv).dry_run
    log(f"paper-tutor {git_sha()}: refresh{' (dry run)' if dry_run else ''}")
    start = time.perf_counter()
    try:
        config = load_config()
        _, model_cfg = active_model(config)
        result = refresh(config, load_model(model_cfg), model_cfg, dry_run=dry_run)
    except Exception as error:  # noqa: BLE001 - every failure must reach Telegram
        log(redact(traceback.format_exc()))
        notify(failure_message(error))
        return 1
    log(json.dumps(result, indent=2))
    log(f"{time.perf_counter() - start:.0f} s")
    return 0 if notify(message(result)) else 1


def cli():
    sys.exit(main(sys.argv[1:]))


if __name__ == "__main__":
    cli()
