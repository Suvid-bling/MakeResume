"""GET /api/v1/config/style-samples"""

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.prompts.style import STYLE_SAMPLES


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
@patch("app.routers.config._load_config")
async def test_get_style_samples_returns_registry_and_content_language(
    mock_load, client
):
    mock_load.return_value = {"content_language": "zh"}
    async with client:
        resp = await client.get("/api/v1/config/style-samples")

    assert resp.status_code == 200
    body = resp.json()
    assert body["content_language"] == "zh"
    assert len(body["samples"]) == len(STYLE_SAMPLES)
    ids = {s["id"] for s in body["samples"]}
    assert ids == {s.id for s in STYLE_SAMPLES}
    first = body["samples"][0]
    assert first["text"].strip()
    assert first["label"].strip()
    assert first["language"]


@pytest.mark.asyncio
@patch("app.routers.config._load_config")
async def test_get_style_samples_defaults_language_when_unset(mock_load, client):
    mock_load.return_value = {}
    async with client:
        resp = await client.get("/api/v1/config/style-samples")

    assert resp.status_code == 200
    assert resp.json()["content_language"] == "en"
