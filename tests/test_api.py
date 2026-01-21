
from fastapi.testclient import TestClient
from src.project.backend import app

client = TestClient(app)

def test_root_health_check():
	response = client.get("/")
	assert response.status_code == 200
	data = response.json()
	assert data["message"] == "OK"
	assert data["status-code"] == 200
