"""Fixture data loader for the UniTSFM test suite.

All fixture files are stored relative to this directory.  Use the helpers
below in tests instead of hard-coding paths::

    from tests.fixtures import load_csv, load_json, FIXTURES_DIR

    df = load_csv("sample_daily.csv")
    data = load_json("precomputed_results.json")
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

FIXTURES_DIR = Path(__file__).resolve().parent
"""Absolute path to the fixtures directory."""


def load_csv(name: str, **kwargs: Any) -> pd.DataFrame:
    """Load a CSV fixture file as a DataFrame.

    Parameters
    ----------
    name:
        File name (e.g. ``"sample_daily.csv"``).
    **kwargs:
        Extra arguments forwarded to ``pd.read_csv()``.

    Returns
    -------
    pd.DataFrame
    """
    path = FIXTURES_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")
    return pd.read_csv(path, **kwargs)


def load_json(name: str) -> dict | list:
    """Load a JSON fixture file.

    Parameters
    ----------
    name:
        File name (e.g. ``"precomputed_results.json"``).

    Returns
    -------
    dict or list
    """
    path = FIXTURES_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_pickle(name: str) -> Any:
    """Load a pickled fixture file.

    Parameters
    ----------
    name:
        File name (e.g. ``"precomputed_forecasts.pkl"``).

    Returns
    -------
    Any
    """
    path = FIXTURES_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")
    return pickle.loads(path.read_bytes())


def fixture_path(name: str) -> Path:
    """Return the absolute path to a fixture file.

    Useful when a test function needs a ``pathlib.Path`` or a string path
    (e.g. for CLI tests that accept ``--data`` arguments).
    """
    path = FIXTURES_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")
    return path


def list_fixtures(suffix: str | None = None) -> list[Path]:
    """List all fixture files, optionally filtered by suffix.

    Parameters
    ----------
    suffix:
        If provided, only return files ending with this string
        (e.g. ``".csv"``, ``".json"``).

    Returns
    -------
    list[Path]
    """
    if suffix:
        return sorted(FIXTURES_DIR.glob(f"*{suffix}"))
    return sorted(FIXTURES_DIR.iterdir())
