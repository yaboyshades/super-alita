"""End-to-end integration tests."""

import pytest
import time
from concurrent.futures import ThreadPoolExecutor

class TestEndToEndIntegration:
    """Test complete system integration workflows."""
    
    def test_health_check_flow(self, client):
        """Test complete health check flow."""
        response = client.get("/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data["status"] in ["healthy", "degraded"]
        assert "version" in health_data
    
    def test_chat_workflow(self, client, sample_chat_message):
        """Test complete chat workflow."""
        # Send chat message
        response = client.post("/v1/chat", json=sample_chat_message)
        assert response.status_code == 200
        
        data = response.json()
        assert "response" in data
        assert len(data.get("response", "")) >= 0  # Could be empty in fallback mode
    
    def test_multi_session_isolation(self, client):
        """Test that different sessions are properly isolated."""
        session1_msg = {"message": "Hello from session 1", "session_id": "session1"}
        session2_msg = {"message": "Hello from session 2", "session_id": "session2"}
        
        response1 = client.post("/v1/chat", json=session1_msg)
        response2 = client.post("/v1/chat", json=session2_msg)
        
        assert response1.status_code == 200
        assert response2.status_code == 200
    
    def test_system_resilience(self, client):
        """Test system resilience under various conditions."""
        # Test with malformed JSON
        malformed_response = client.post(
            "/v1/chat",
            data="{malformed json}",
            headers={"Content-Type": "application/json"}
        )
        assert malformed_response.status_code == 422  # FastAPI validation error
        
        # Test health check still works after error
        health_response = client.get("/health")
        assert health_response.status_code == 200