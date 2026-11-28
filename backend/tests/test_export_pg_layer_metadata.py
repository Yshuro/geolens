"""Which ogr2ogr exports read GDAL layer metadata from the database.

GDAL's PostgreSQL driver reads layer metadata from the ogr_system_tables
schema when it exists, and warns when the runtime role has no USAGE on it.
Formats that write none of that metadata pass
--config OGR_PG_ENABLE_METADATA NO. GeoPackage, FlatGeobuf and GeoJSON carry
it into the file, so they keep the lookup.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.processing.export.ogr import (
    _NO_LAYER_METADATA_FORMATS,
    FORMAT_MAP,
    run_ogr2ogr_export,
)


async def _exported_config(tmp_path, format_key: str) -> dict[str, str]:
    """The --config options run_ogr2ogr_export passes for ``format_key``."""
    argv: list[str] = []

    async def _fake_exec(*args, **kwargs):
        argv.extend(args)
        if len(args) > 3:
            Path(args[3]).touch()
        proc = MagicMock()
        proc.returncode = 0
        return proc

    async def _fake_communicate(proc, timeout, tool_name):
        return (b"", b"")

    with (
        patch("asyncio.create_subprocess_exec", side_effect=_fake_exec),
        patch(
            "app.processing.export.ogr._communicate_with_timeout",
            new=_fake_communicate,
        ),
    ):
        await run_ogr2ogr_export(
            "test_table",
            str(tmp_path / f"out.{format_key}"),
            FORMAT_MAP[format_key]["driver"],
            schema="data",
            format_key=format_key,
        )

    return {argv[i + 1]: argv[i + 2] for i, arg in enumerate(argv) if arg == "--config"}


@pytest.mark.anyio
@pytest.mark.parametrize("format_key", ["csv", "kml", "kmz", "pmtiles", "shp"])
async def test_export_without_layer_metadata_skips_the_lookup(tmp_path, format_key):
    config = await _exported_config(tmp_path, format_key)
    assert config.get("OGR_PG_ENABLE_METADATA") == "NO"


@pytest.mark.anyio
@pytest.mark.parametrize("format_key", ["fgb", "geojson", "gpkg"])
async def test_export_that_carries_layer_metadata_keeps_the_lookup(
    tmp_path, format_key
):
    config = await _exported_config(tmp_path, format_key)
    assert "OGR_PG_ENABLE_METADATA" not in config


def test_every_ogr_format_is_placed():
    """A new format has to be put on one side or the other."""
    carries = set(FORMAT_MAP) - _NO_LAYER_METADATA_FORMATS
    assert carries == {"fgb", "geojson", "gpkg"}
