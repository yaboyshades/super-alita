"""Performance and load testing."""

import pytest
import time
from concurrent.futures import ThreadPoolExecutor

class TestLoadPerformance:
    """Performance and load testing."""
    
    def test_concurrent_health_checks(self, client):
        """Test handling multiple concurrent health check requests."""
        num_requests = 20
        
        def send_health_request():
            response = client.get("/health")
            return response.status_code
        
        # Send requests concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            start_time = time.time()
            results = list(executor.map(lambda _: send_health_request(), range(num_requests)))
            end_time = time.time()
        
        # Check all requests succeeded
        assert all(code == 200 for code in results)
        
        # Performance check
        total_time = end_time - start_time
        assert total_time < 5.0  # Should complete within 5 seconds