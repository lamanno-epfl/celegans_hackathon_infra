"""Global configuration. All values env-overridable."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass
class ScoringConfig:
    registration_weight: float = _env_float("REGISTRATION_WEIGHT", 0.8)
    integration_weight: float = _env_float("INTEGRATION_WEIGHT", 0.2)
    registration_threshold: float = _env_float("REGISTRATION_THRESHOLD", 0.3)
    collapse_std_threshold: float = _env_float("COLLAPSE_STD_THRESHOLD", 1e-6)
    integration_k_folds: int = _env_int("INTEGRATION_K_FOLDS", 5)
    random_seed: int = _env_int("SCORING_SEED", 42)
    min_embedding_dim: int = 64
    max_embedding_dim: int = 2048


@dataclass
class HarborConfig:
    url: str = _env("HARBOR_URL", "https://registry.competition.org")
    admin_user: str = _env("HARBOR_ADMIN_USER", "admin")
    admin_password: str = _env("HARBOR_ADMIN_PASSWORD", "Harbor12345")
    webhook_secret: str = _env("HARBOR_WEBHOOK_SECRET", "changeme")


@dataclass
class SMTPConfig:
    host: str = _env("SMTP_HOST", "localhost")
    port: int = _env_int("SMTP_PORT", 25)
    user: str = _env("SMTP_USER", "")
    password: str = _env("SMTP_PASSWORD", "")
    from_address: str = _env("SMTP_FROM", "competition@example.org")
    use_tls: bool = _env("SMTP_TLS", "false").lower() == "true"
    # If true, emails are logged to stdout instead of sent (for dev).
    dry_run: bool = _env("SMTP_DRY_RUN", "true").lower() == "true"


@dataclass
class OrchestratorConfig:
    database_url: str = _env("DATABASE_URL", f"sqlite:///{ROOT / 'runtime' / 'orchestrator.db'}")
    api_key: str = _env("ORCHESTRATOR_API_KEY", "dev-api-key")
    queue_dir: Path = Path(_env("QUEUE_DIR", str(ROOT / "runtime" / "queue")))
    work_dir: Path = Path(_env("WORK_DIR", str(ROOT / "runtime" / "work")))
    eval_timeout_seconds: int = _env_int("EVAL_TIMEOUT", 120 * 60)
    max_submissions_default: int = _env_int("MAX_SUBMISSIONS", 11)


@dataclass
class DataConfig:
    root: Path = Path(_env("DATA_ROOT", str(ROOT / "data")))
    simulated_public_ratio: float = _env_float("SIM_PUBLIC_RATIO", 0.7)
    real_public_ratio: float = _env_float("REAL_PUBLIC_RATIO", 0.8)
    atlas_dir: Path = Path(_env("ATLAS_DIR", str(ROOT / "data" / "reference_4d")))


@dataclass
class Config:
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    harbor: HarborConfig = field(default_factory=HarborConfig)
    smtp: SMTPConfig = field(default_factory=SMTPConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    data: DataConfig = field(default_factory=DataConfig)


CONFIG = Config()
