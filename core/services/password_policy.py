"""One server-side password policy for signup, changes, and recovery."""
import re

MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128
PASSWORD_HELP = "Use 8–128 characters. A unique passphrase of several unrelated words works well. Avoid common passwords and your account details."
_COMMON = re.compile(
    r"(?:password|p@ssw[0o]rd|qwerty|letmein|welcome|admin|examly|iloveyou|changeme|1234567890?)[\d\W_]*",
    re.IGNORECASE,
)


def validate_password(password: str, *, username: str = "", email: str = "") -> None:
    if len(password.strip()) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters long.")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at most {MAX_PASSWORD_LENGTH} characters long.")
    compact = password.casefold().strip()
    sequence = compact in "0123456789" * 13 or compact in "abcdefghijklmnopqrstuvwxyz" * 5
    repeated = re.fullmatch(r"(.{1,12})\1{2,}", compact, re.DOTALL)
    if not compact or len(set(compact)) < 4 or sequence or repeated or _COMMON.fullmatch(compact):
        raise ValueError("Choose a less predictable password. Try several unrelated words.")
    if compact in {value.casefold().strip() for value in (username, email, email.split("@")[0]) if value}:
        raise ValueError("Your password must be different from your username and email address.")


def needs_password_upgrade(password: str, **identity) -> bool:
    try:
        validate_password(password, **identity)
    except ValueError:
        return True
    return False
