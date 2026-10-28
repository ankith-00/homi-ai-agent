import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Homi AI agent API")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the API to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the API to")
    parser.add_argument("--reload", action="store_true", help="Enable development auto-reload")
    args = parser.parse_args()

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)
