$env:PYTHONPATH="src"; poetry run uvicorn main:app --reload --host 0.0.0.0 --port 8000

curl -X POST "http://localhost:8000/ask/" -F "query=What is the value of one bearing?"

ssh -i 111.237.107.89 -p 54850