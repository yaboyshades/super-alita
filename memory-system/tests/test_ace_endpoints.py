from fastapi.testclient import TestClient

from src.app import app

client = TestClient(app)


def _seed_context() -> None:
    client.post(
        "/capture",
        json=[
            {
                "role": "user",
                "content": "Remember that I enjoy solar energy projects and rooftop panels.",
                "meta": {"topic": "energy"},
            },
            {
                "role": "user",
                "content": "Sometimes I switch to tea even though I prefer coffee.",
                "meta": {"topic": "beverage"},
            },
        ],
    )


class TestACEEndpoints:
    def test_context_evolve_endpoint(self) -> None:
        _seed_context()
        response = client.post(
            "/context/evolve",
            params={"q": "solar", "k": 5, "budget": 400},
            json={"low_confidence": True},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["citations"]
        assert payload["provenance"].get("ace_enabled") == "true"

    def test_strategy_evaluation(self) -> None:
        _seed_context()
        client.post(
            "/context/evolve",
            params={"q": "coffee", "k": 5, "budget": 400},
            json={"contradictions": [{"claim": "coffee"}]},
        )
        feedback = {"response_quality": 0.8, "contradictions_resolved": True, "clarity": 0.9}
        response = client.post("/ace/strategies/evaluate", json=feedback)
        assert response.status_code == 200
        record = response.json()
        assert record["cycle"] >= 1
        assert record["strategies"] is not None

    def test_evolution_history(self) -> None:
        _seed_context()
        client.post(
            "/context/evolve",
            params={"q": "history", "k": 3, "budget": 300},
            json={"low_confidence": True},
        )
        client.post(
            "/ace/strategies/evaluate",
            json={"response_quality": 0.7, "contradictions_resolved": False},
        )
        history = client.get("/ace/evolution/history")
        assert history.status_code == 200
        payload = history.json()
        assert "history" in payload
        assert "total_cycles" in payload
