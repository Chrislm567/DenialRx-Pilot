from pathlib import Path

import pytest
import yaml


@pytest.mark.skipif(not Path("docker-compose.yml").exists(), reason="docker-compose missing")
def test_docker_compose_includes_api_service():
    with Path("docker-compose.yml").open() as handle:
        compose = yaml.safe_load(handle)
    assert "services" in compose
    assert "api" in compose["services"]
    api_service = compose["services"]["api"]
    assert api_service["command"].startswith("uvicorn app.main:app")
