# API Client tests
from src.abilities.api_client import create_api_client


def test_client_creation():
    client = create_api_client("https://api.test.com")
    assert client.base_url == "https://api.test.com"
