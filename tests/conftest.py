import pytest
from fastapi.testclient import TestClient
from src.app.main import app  

client = TestClient(app)