"""SMTP email service with Jinja2 templates."""
from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict

from jinja2 import Environment, FileSystemLoader, select_autoescape

from config import CONFIG

log = logging.getLogger(__name__)

_TEMPLATES = Path(__file__).parent / "templates"
_env = Environment(
    loader=FileSystemLoader(_TEMPLATES),
    autoescape=select_autoescape(["html"]),
)

SUBJECTS: Dict[str, str] = {
    "submission_received": "Submission received",
    "evaluation_started": "Evaluation started",
    "evaluation_complete": "Evaluation complete",
    "evaluation_failed": "Evaluation failed",
    "submission_limit_reached": "Submission limit reached",
    "validation_error": "Container validation error",
}


def _render(template_name: str, context: Dict[str, Any]) -> tuple[str, str]:
    text = _env.get_template(f"{template_name}.txt").render(**context)
    try:
        html = _env.get_template(f"{template_name}.html").render(**context)
    except Exception:
        html = None
    return text, html


def send_email(to_address: str, template: str, context: Dict[str, Any]) -> None:
    """Render and send an email. Uses SMTP unless SMTP_DRY_RUN is true."""
    subject = SUBJECTS.get(template, template)
    text, html = _render(template, context)

    if CONFIG.smtp.dry_run:
        log.info("[DRY-RUN EMAIL] to=%s subject=%s\n%s", to_address, subject, text)
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = CONFIG.smtp.from_address
    msg["To"] = to_address
    msg.set_content(text)
    if html:
        msg.add_alternative(html, subtype="html")

    smtp_cls = smtplib.SMTP_SSL if CONFIG.smtp.use_tls and CONFIG.smtp.port == 465 else smtplib.SMTP
    with smtp_cls(CONFIG.smtp.host, CONFIG.smtp.port) as s:
        if CONFIG.smtp.use_tls and CONFIG.smtp.port != 465:
            s.starttls()
        if CONFIG.smtp.user:
            s.login(CONFIG.smtp.user, CONFIG.smtp.password)
        s.send_message(msg)
