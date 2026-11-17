# Development web client

Start the backend from the repository root with `docker compose up --build -d`,
then serve this folder with `python -m http.server 5500 --bind 127.0.0.1 --directory frontend`.
Open http://127.0.0.1:5500 and connect to the API at http://127.0.0.1:8000.
Use an appropriate tenant credential from the private `.env` credential map.
Reader credentials can query; writer/admin credentials can also upload and ingest.

The client supports uploads, automatic ingestion, document listing, answers,
source IDs, and a retrieval ranking score. The score is not answer accuracy.
CORS permits only configured origins. The client stores its credential in browser
localStorage as a development convenience; use the organization's identity flow
for a deployed production UI.

Browser integration tests are in `frontend/e2e`. `run.sh` boots an isolated local
hash-embedding backend for the default offline test. The test can also target an
already running production API using `APP_URL`, `API_URL`, and `API_KEY`.
