## You access anything in conftest.py without importing it. Here toy usage of client.
def test_root(client):
    re = client.get("/")
    assert re.status_code == 200
    assert re.json() == {"mlops_mini": "This is a small end-to-end AI project for you!"}
    