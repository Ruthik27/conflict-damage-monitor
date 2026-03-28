"""
db.py — PostGIS storage for inference results (optional).

When DATABASE_URL is set, results are persisted to a PostGIS table.
If unset or the connection fails, the API silently skips persistence
so the server works without a database in development.

Schema:
    CREATE TABLE damage_results (
        job_id      TEXT PRIMARY KEY,
        model_type  TEXT,
        pre_path    TEXT,
        post_path   TEXT,
        created_at  TIMESTAMPTZ DEFAULT now(),
        class_stats JSONB,
        geom        geometry(MultiPolygon, 4326)
    );
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_engine = None   # lazy initialised


def _get_engine():
    global _engine
    if _engine is not None:
        return _engine

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return None

    try:
        from sqlalchemy import create_engine
        _engine = create_engine(db_url, pool_pre_ping=True)
        # Ensure table exists
        _ensure_table(_engine)
        logger.info("PostGIS engine connected: %s", db_url.split("@")[-1])
    except Exception as exc:
        logger.warning("PostGIS unavailable — results will not be persisted: %s", exc)
        _engine = None

    return _engine


def _ensure_table(engine) -> None:
    from sqlalchemy import text
    ddl = """
    CREATE TABLE IF NOT EXISTS damage_results (
        job_id      TEXT PRIMARY KEY,
        model_type  TEXT,
        pre_path    TEXT,
        post_path   TEXT,
        created_at  TIMESTAMPTZ DEFAULT now(),
        class_stats JSONB,
        geom        geometry(MultiPolygon, 4326)
    );
    """
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
        conn.execute(text(ddl))


def save_result(
    job_id: str,
    model_type: str,
    pre_path: str,
    post_path: str,
    class_stats: list[dict],
    geojson: dict[str, Any] | None,
) -> bool:
    """Persist a result to PostGIS. Returns True on success, False if skipped."""
    engine = _get_engine()
    if engine is None:
        return False

    try:
        from sqlalchemy import text

        # Build a MultiPolygon WKT from the GeoJSON FeatureCollection
        geom_sql = "NULL"
        if geojson and geojson.get("features"):
            geoms_wkt = []
            for feat in geojson["features"]:
                geom = feat["geometry"]
                if geom["type"] == "Polygon":
                    coords = geom["coordinates"][0]
                    ring = ", ".join(f"{c[0]} {c[1]}" for c in coords)
                    geoms_wkt.append(f"(({ring}))")
            if geoms_wkt:
                multi = "MULTIPOLYGON(" + ", ".join(geoms_wkt) + ")"
                geom_sql = f"ST_GeomFromText('{multi}', 4326)"

        sql = text(f"""
            INSERT INTO damage_results
                (job_id, model_type, pre_path, post_path, class_stats, geom)
            VALUES
                (:job_id, :model_type, :pre_path, :post_path, :class_stats, {geom_sql})
            ON CONFLICT (job_id) DO NOTHING;
        """)
        with engine.begin() as conn:
            conn.execute(sql, {
                "job_id":      job_id,
                "model_type":  model_type,
                "pre_path":    pre_path,
                "post_path":   post_path,
                "class_stats": json.dumps(class_stats),
            })
        return True
    except Exception as exc:
        logger.error("Failed to save result %s: %s", job_id, exc)
        return False


def get_result(job_id: str) -> dict[str, Any] | None:
    """Fetch a stored result by job_id."""
    engine = _get_engine()
    if engine is None:
        return None

    try:
        from sqlalchemy import text
        sql = text("""
            SELECT job_id, model_type, pre_path, post_path,
                   created_at, class_stats,
                   ST_AsGeoJSON(geom)::jsonb AS geom
            FROM damage_results
            WHERE job_id = :job_id;
        """)
        with engine.connect() as conn:
            row = conn.execute(sql, {"job_id": job_id}).fetchone()
        if row is None:
            return None
        return dict(row._mapping)
    except Exception as exc:
        logger.error("Failed to fetch result %s: %s", job_id, exc)
        return None
