from pathlib import Path
from typing import Any, Dict

from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.models import AppealDraft, AppealSubmission

TEMPLATES_PATH = Path(__file__).resolve().parent.parent / "templates"

env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_PATH)),
    autoescape=select_autoescape(enabled_extensions=("jinja",)),
)


def render_denial_letter(draft: AppealDraft) -> str:
    template = env.get_template("appeals/denial_letter.jinja")
    return template.render(draft=draft)


def render_submission_email(appeal_id: str, draft: AppealSubmission) -> Dict[str, Any]:
    """Render MJML email and provide raw + metadata for downstream mailers."""
    template = env.get_template("appeals/appeal_email.mjml")
    mjml_body = template.render(appeal_id=appeal_id, draft=draft)
    subject = f"Appeal {appeal_id} submitted"
    return {
        "appeal_id": appeal_id,
        "subject": subject,
        "mjml": mjml_body,
    }
