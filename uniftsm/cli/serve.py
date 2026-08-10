"""``uniftsm serve`` — start a lightweight forecasting API server."""

from __future__ import annotations

import logging

import click
import pandas as pd

logger = logging.getLogger("uniftsm.cli.serve")


@click.command(name="serve")
@click.option("--model", "-m", default="chronos", help="Model name to serve.")
@click.option(
    "--model-size",
    default="small",
    help="Model size / variant (e.g., 'small', 'base').",
)
@click.option("--host", default="127.0.0.1", help="Host to bind the server to.")
@click.option("--port", "-p", default=8080, type=int, help="Port to bind the server to.")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Verbose output.")
def serve(
    model: str,
    model_size: str,
    host: str,
    port: int,
    verbose: bool,
) -> None:
    """Start a REST API server for forecasting.

    The server exposes a single endpoint:

        POST /forecast
        {
            "series": [1.0, 2.0, 3.0, ...],
            "horizon": 24
        }
        → { "mean": [...], "std": [...] }

    Examples:

        uniftsm serve --model chronos --port 8080

        uniftsm serve --model moirai --model-size base
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        from uniftsm.core.registry import registry

        model_cls = registry.get(model)
    except ValueError as exc:
        click.echo(f"Error: {exc}")
        return

    click.echo(f"Loading model '{model}' (size: {model_size})...")
    forecaster = model_cls(model_size=model_size, device="auto")
    click.echo(f"Model '{model}' loaded. Ready for inference.")

    click.echo(f"Starting API server at http://{host}:{port}...")
    click.echo("  POST /forecast  —  Generate forecasts")
    click.echo("  GET /health    —  Health check")

    try:
        import http.server
        import json

        class ForecastHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: str) -> None:
                logger.info(fmt, *args)

            def do_GET(self) -> None:
                if self.path == "/health":
                    self._respond_json({"status": "ok", "model": model})
                else:
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b'{"error": "Not found"}')

            def do_POST(self) -> None:
                if self.path != "/forecast":
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b'{"error": "Not found"}')
                    return

                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                try:
                    data = json.loads(body)
                except json.JSONDecodeError:
                    self._respond_json({"error": "Invalid JSON"}, status=400)
                    return

                series = data.get("series")
                horizon = data.get("horizon", 24)
                if not series or not isinstance(series, list):
                    self._respond_json({"error": "Missing or invalid 'series'"}, status=400)
                    return

                try:
                    y = pd.Series(series)
                    forecaster.fit(y)
                    pred = forecaster.predict(horizon, return_quantiles=False)
                    self._respond_json(
                        {
                            "mean": pred["mean"].tolist(),
                            "std": pred["std"].tolist(),
                            "horizon": horizon,
                            "model": model,
                        }
                    )
                except Exception as exc:
                    self._respond_json({"error": str(exc)}, status=500)

            def _respond_json(self, data: dict, status: int = 200) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())

        server = http.server.HTTPServer((host, port), ForecastHandler)
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("\nServer stopped.")
    except Exception as exc:
        click.echo(f"Server error: {exc}")
