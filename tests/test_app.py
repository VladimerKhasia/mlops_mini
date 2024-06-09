def test_root(client):
    re = client.get("/")
    assert re.status_code == 200
    assert re.json() == {"mlops_mini": "This is a small end-to-end AI project for you!"}
    
# def test_mocked_root(mocker, client):
#     mocker.patch('src.app.agent_service.AutoModelForCausalLM.from_pretrained', return_value=None)
#     mocker.patch('src.app.agent_service.conversational_agent', return_value=None)
#     re = client.get("/")
#     assert re.status_code == 200
#     assert re.json() == {"mlops_mini": "This is a small end-to-end AI project for you!"}    

## use test_mocked_root if you add pytest-mock in .github/workflows/main.yaml file:
      # - name: run pytest tests
      #   run: |
      #     pip install pytest pytest-mock
      #     pytest -v -s -x