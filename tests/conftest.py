from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import psycopg2
import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCKER_COMPOSE = ["docker", "compose", "-f", str(PROJECT_ROOT / "docker-compose.yml")]
DEFAULT_DSN = "postgresql://ingest:ingest@localhost:6543/ingest"


@pytest.fixture(scope="session")
def postgres_service():
    external_dsn = os.environ.get("TEST_DSN")
    dsn = external_dsn or DEFAULT_DSN
    if external_dsn:
        _wait_for_dsn(dsn)
        yield dsn
        return

    if shutil.which("docker") is None:
        pytest.skip("Docker is not available in the test environment")

    subprocess.run(DOCKER_COMPOSE + ["up", "-d", "postgres"], check=True)
    try:
        _wait_for_dsn(dsn)
        yield dsn
    finally:
        subprocess.run(DOCKER_COMPOSE + ["down", "-v"], check=True)


def _wait_for_dsn(dsn: str) -> None:
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            with psycopg2.connect(dsn):
                return
        except psycopg2.OperationalError:
            time.sleep(1)
    raise RuntimeError("Postgres service did not become healthy")


@pytest.fixture(scope="session")
def dsn(postgres_service: str) -> str:
    return postgres_service
