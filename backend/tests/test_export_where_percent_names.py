"""The export row count and GeoParquet read a filter holding ``%(name)s`` text as written."""

import uuid

import pytest
from sqlalchemy import text

_LABEL = "%(label)s"
_COLUMNS = [
    {"name": "gid", "type": "integer"},
    {"name": "name", "type": "text"},
    {"name": _LABEL, "type": "text"},
]
_NAMES = [
    "%(limit)s",
    "%(minx)s",
    "%(__cap)s",
    "%(former)school",
    "100%",
    "a%%b",
    "a:b",
    "plain",
    "\\%(limit)s:\\",
    "__[POSTCOMPILE_limit]",
]
_WORLD = [-180.0, -90.0, 180.0, 90.0]

# (filter, rows it selects as written, whether the request carries a bbox)
_CASES = [
    ("name = '%(limit)s'", 1, False),
    ("'%(limit)s' <> '$1'", 10, False),
    ("name = '%(minx)s'", 1, True),
    ("name = '%(__cap)s'", 1, True),
    ("name LIKE '%(former)school%'", 1, False),
    ("name LIKE '%(%' OR name = 'a%%b'", 6, False),
    ("name LIKE '100%'", 1, False),
    ("\"%(label)s\" = '%(limit)s'", 1, False),
    ("name = 'a:b' OR name = '%(limit)s'", 2, False),
    ("name = '\\%(limit)s:\\'", 1, True),
    ("name = '__[POSTCOMPILE_limit]'", 1, False),
]
_IDS = [f"case{i}" for i in range(len(_CASES))]


@pytest.fixture
async def percent_table(test_db_session, monkeypatch, tmp_path):
    from app.processing.export import service as export_service

    monkeypatch.setattr(export_service.settings, "upload_staging_dir", str(tmp_path))
    table = f"exp_pct_{uuid.uuid4().hex[:12]}"
    # Through the driver: text() can't spell this column's name.
    conn = await test_db_session.connection()
    await conn.exec_driver_sql(
        f'CREATE TABLE data.{table} (gid serial PRIMARY KEY, name text, "%(label)s" '
        "text, geom geometry(Point, 4326), geom_4326 geometry(Point, 4326) "
        "GENERATED ALWAYS AS (geom) STORED)"
    )
    for i, name in enumerate(_NAMES):
        await conn.exec_driver_sql(
            f'INSERT INTO data.{table} (name, "%(label)s", geom) VALUES '
            "($1, $1, ST_SetSRID(ST_MakePoint($2, 10), 4326))",
            (name, float(i)),
        )
    await test_db_session.commit()
    yield table
    await test_db_session.rollback()
    await test_db_session.execute(text(f"DROP TABLE IF EXISTS data.{table}"))
    await test_db_session.commit()


def _render(where: str) -> str:
    """The filter as ogr2ogr receives it."""
    from app.processing.export.service import validate_where_clause

    return validate_where_clause(where, _COLUMNS)


async def _driver_names(session, table: str, where: str, bbox: bool) -> list[str]:
    """The names the render selects with no SQLAlchemy compile, as ogr2ogr reads it.

    ogr2ogr turns standard_conforming_strings on for its connection. The read
    does the same in a transaction of its own, which ends before the export
    runs so the export has to set the mode itself.
    """
    envelope = " AND geom_4326 && ST_MakeEnvelope(-180, -90, 180, 90, 4326)"
    conn = await session.connection()
    await conn.exec_driver_sql("SET LOCAL standard_conforming_strings = on")
    result = await conn.exec_driver_sql(
        f"SELECT name FROM data.{table} WHERE ({_render(where)})"
        + (envelope if bbox else "")
    )
    names = sorted(result.scalars().all())
    await session.rollback()
    return names


@pytest.mark.anyio
@pytest.mark.parametrize(("where", "expected", "bbox"), _CASES, ids=_IDS)
async def test_the_count_reads_the_filter_as_written(
    test_db_session, percent_table, where, expected, bbox
):
    from app.processing.export import selection

    names = await _driver_names(test_db_session, percent_table, where, bbox)
    assert len(names) == expected
    count = await selection._count_selected_features(
        test_db_session,
        table_name=percent_table,
        where=where,
        column_info=_COLUMNS,
        bbox=_WORLD if bbox else None,
        has_geometry=True,
        schema="data",
    )
    assert count == expected


_LEGACY_ESCAPES = text("SET standard_conforming_strings = off")
_TWO_ROWS = "name LIKE '%(former)school%' OR name = '%(limit)s'"


@pytest.fixture
async def legacy_string_escapes(test_db_session):
    """Each test sets the mode itself, on the connection it is about to use."""
    yield
    await test_db_session.rollback()
    await test_db_session.execute(text("RESET standard_conforming_strings"))
    await test_db_session.commit()


@pytest.mark.anyio
async def test_the_count_reads_the_filter_as_written_without_conforming_strings(
    test_db_session, percent_table, legacy_string_escapes
):
    from app.processing.export import selection

    await test_db_session.execute(_LEGACY_ESCAPES)
    count = await selection._count_selected_features(
        test_db_session,
        table_name=percent_table,
        where=_TWO_ROWS,
        column_info=_COLUMNS,
        bbox=None,
        has_geometry=True,
        schema="data",
    )
    assert count == 2


@pytest.mark.anyio
async def test_the_geoparquet_count_reads_the_filter_without_conforming_strings(
    test_db_session, percent_table, legacy_string_escapes, monkeypatch
):
    from app.processing.export import parquet

    monkeypatch.setattr(parquet, "_MAX_EXPORT_FEATURES", 2)
    await test_db_session.execute(_LEGACY_ESCAPES)
    await parquet.plan_parquet_export(
        test_db_session, percent_table, schema="data", where=_TWO_ROWS
    )
    monkeypatch.setattr(parquet, "_MAX_EXPORT_FEATURES", 1)
    with pytest.raises(parquet.ExportTooLargeError):
        await parquet.plan_parquet_export(
            test_db_session, percent_table, schema="data", where=_TWO_ROWS
        )


@pytest.mark.anyio
async def test_the_geoparquet_stream_reads_the_filter_without_conforming_strings(
    test_db_session, percent_table, legacy_string_escapes
):
    import pyarrow.parquet as pq

    from app.processing.export.parquet import export_parquet, plan_parquet_export

    plan = await plan_parquet_export(
        test_db_session, percent_table, schema="data", where=_TWO_ROWS
    )
    # The stream runs in a transaction of its own, which may be on another
    # connection, so the mode is set there and not before the plan.
    await test_db_session.commit()
    await test_db_session.execute(_LEGACY_ESCAPES)
    path, _filename, _media_type = await export_parquet(
        test_db_session, percent_table, "Percent", schema="data", plan=plan
    )
    assert sorted(pq.read_table(path).column("name").to_pylist()) == [
        "%(former)school",
        "%(limit)s",
    ]


@pytest.mark.parametrize(
    "quoted",
    [
        "'%(limit)s'",
        '"%(label)s"',
        "'__[POSTCOMPILE_limit]'",
        '"__[SCHEMA_data]"',
    ],
)
def test_no_text_the_compiler_rewrites_is_left_in_a_quoted_run(quoted):
    from app.processing.export.service import sql_for_text

    out = sql_for_text(quoted)
    assert "%(" not in out
    assert "__[" not in out


@pytest.mark.anyio
@pytest.mark.parametrize(("where", "expected", "bbox"), _CASES, ids=_IDS)
async def test_the_geoparquet_count_reads_the_filter_as_written(
    test_db_session, percent_table, where, expected, bbox, monkeypatch
):
    from app.processing.export import parquet

    names = await _driver_names(test_db_session, percent_table, where, bbox)
    assert len(names) == expected
    # The plan refuses a selection over the cap, so the count is exactly
    # ``expected`` when a cap of ``expected`` passes and one below refuses.
    monkeypatch.setattr(parquet, "_MAX_EXPORT_FEATURES", expected)
    await parquet.plan_parquet_export(
        test_db_session,
        percent_table,
        schema="data",
        bbox=_WORLD if bbox else None,
        where=where,
    )
    monkeypatch.setattr(parquet, "_MAX_EXPORT_FEATURES", expected - 1)
    with pytest.raises(parquet.ExportTooLargeError):
        await parquet.plan_parquet_export(
            test_db_session,
            percent_table,
            schema="data",
            bbox=_WORLD if bbox else None,
            where=where,
        )


@pytest.mark.anyio
@pytest.mark.parametrize(("where", "expected", "bbox"), _CASES, ids=_IDS)
async def test_the_geoparquet_stream_reads_the_filter_as_written(
    test_db_session, percent_table, where, expected, bbox
):
    import pyarrow.parquet as pq

    from app.processing.export.parquet import export_parquet, plan_parquet_export

    driver_names = await _driver_names(test_db_session, percent_table, where, bbox)
    assert len(driver_names) == expected
    plan = await plan_parquet_export(
        test_db_session,
        percent_table,
        schema="data",
        bbox=_WORLD if bbox else None,
        where=where,
    )
    await test_db_session.commit()
    path, _filename, _media_type = await export_parquet(
        test_db_session, percent_table, "Percent", schema="data", plan=plan
    )
    written = pq.read_table(path)
    assert sorted(written.column("name").to_pylist()) == driver_names
    assert sorted(written.column(_LABEL).to_pylist()) == driver_names
