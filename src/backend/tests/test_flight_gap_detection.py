import io
import zipfile

import pytest
from shapely.geometry import MultiPolygon, box, mapping
from shapely.ops import transform

from drone_flightplan.drone_type import DroneType

from app.images.flight_gap_identification import (
    _detect_sparse_coverage_gap,
    _generate_flightplan_for_geometry,
    identify_flight_gaps,
)
from app.images.image_footprints import inverse_projector


def assert_is_valid_flightplan(buffer: bytes):
    """Verifies the result is a valid KMZ bytes containing flight instructions."""

    assert isinstance(buffer, bytes)
    assert buffer.startswith(b"PK")

    with zipfile.ZipFile(io.BytesIO(buffer)) as z:
        filenames = [f.lower() for f in z.namelist()]
        valid_extensions = [".kml", ".wpml", ".plan", ".json"]
        assert any(
            any(extension in f for extension in valid_extensions) for f in filenames
        ), f"No valid flight plan files found in KMZ. Files: {filenames}"


def _point_from_meters(x: float, y: float) -> dict:
    # Build test image GPS points from meter coordinates so the geometry is easy.
    lon, lat = inverse_projector.transform(x, y)
    return {"type": "Point", "coordinates": [lon, lat]}


def _aoi_from_meters(min_x: float, min_y: float, max_x: float, max_y: float):
    # The real code receives lon/lat geometry, so convert this simple meter box back.
    return transform(
        inverse_projector.transform,
        box(min_x, min_y, max_x, max_y),
    )


def test_sparse_coverage_gap_generates_single_gap_flightplan():
    # Images cover only the left side of the task, leaving one clear missing area.
    task_aoi = _aoi_from_meters(-150, -100, 150, 100)
    images = [
        {
            "id": f"left-{idx}",
            "image_location_json": _point_from_meters(-120, y),
            "yaw_deg": 90,
        }
        for idx, y in enumerate([-75, -55, -35, -15, 5, 25, 45, 65])
    ]

    gap = _detect_sparse_coverage_gap(
        task_aoi,
        images,
        DroneType.DJI_MINI_4_PRO,
        average_altitude=60,
    )
    result = _generate_flightplan_for_geometry(
        gap,
        DroneType.DJI_MINI_4_PRO,
        average_altitude=60,
        rotation_angle=90,
    )

    assert gap is not None
    assert gap.geom_type == "Polygon"
    assert result is not None
    assert result["geometry"].geom_type == "Polygon"
    assert_is_valid_flightplan(result["kmz_bytes"])


def test_sparse_coverage_gap_generates_two_gap_flightplan():
    # A flown strip through the middle splits the missing area into top and bottom gaps.
    task_aoi = _aoi_from_meters(-150, -150, 150, 150)
    images = [
        {
            "id": f"middle-{idx}",
            "image_location_json": _point_from_meters(x, 0),
            "yaw_deg": 90,
        }
        for idx, x in enumerate([-125, -105, -85, -65, -45, -25, -5, 15, 35, 55, 75, 95, 115])
    ]

    gap = _detect_sparse_coverage_gap(
        task_aoi,
        images,
        DroneType.DJI_MINI_4_PRO,
        average_altitude=60,
    )
    result = _generate_flightplan_for_geometry(
        gap,
        DroneType.DJI_MINI_4_PRO,
        average_altitude=60,
        rotation_angle=90,
    )

    assert isinstance(gap, MultiPolygon)
    assert len(gap.geoms) == 2
    assert result is not None
    assert isinstance(result["geometry"], MultiPolygon)
    assert_is_valid_flightplan(result["kmz_bytes"])


@pytest.mark.asyncio
async def test_north_wrap_around(db, load_freetown_into_db):
    """
    Ensures circular mean handles a North wrap (359 -> 1) around.
    """
    north_path = []
    for i in range(60):
        # Creating 4 gaps to satisfy MIN_GAP_IMAGES threshold
        if i in [10, 20, 30, 40]:
            continue

        # Oscillating around 359 - 1 degrees.
        if i % 2:
            yaw_degree = 359
        else:
            yaw_degree = 1

        # Ensure 0.2 arc-seconds as >5m threshold
        north_path.append(
            {
                "SourceFile": f"north_{i}.jpg",
                "GPSLatitude": f"8 deg 28' {8.4 + (i * 0.2)}\" N",
                "GPSLongitude": "13 deg 11' 49.2\" W",
                "AbsoluteAltitude": "100",
                "DateTimeOriginal": f"2024:01:01 12:00:{i:02d}",
                "FlightYawDegree": str(yaw_degree),
            }
        )

    project_id, batch_id, task_id = await load_freetown_into_db(
        manual_metadata=north_path
    )
    result = await identify_flight_gaps(db, project_id, task_id)

    assert "type" in result["gap_polygons"]
    assert "images" in result


@pytest.mark.asyncio
async def test_gap_missing_flight_leg(db, load_freetown_into_db):
    """
    Verifies side overlap gaps.
    """
    side_gap_metadata = []
    # 10 legs total:
    # 0-5 are Normal (15m apart)
    # 6-9 are Gaps (400m apart)
    lat_offsets = [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 40.0, 70.0, 100.0, 130.0]
    yaw_degrees = ["90", "270", "90", "270", "90", "270", "90", "270", "90", "270"]

    for leg_idx, (lat_sec, yaw_value) in enumerate(zip(lat_offsets, yaw_degrees)):
        for i in range(15):
            side_gap_metadata.append(
                {
                    "SourceFile": f"leg{leg_idx}_{i}.jpg",
                    "GPSLatitude": f"8 deg 28' {lat_sec}\" N",
                    "GPSLongitude": f"13 deg 11' {45.6 + (i * 0.3)}\" W",
                    "AbsoluteAltitude": "100",
                    "DateTimeOriginal": f"2024:01:01 12:{leg_idx:02d}:{i:02d}",
                    "FlightYawDegree": yaw_value,
                }
            )

    project_id, batch_id, task_id = await load_freetown_into_db(
        manual_metadata=side_gap_metadata
    )
    result = await identify_flight_gaps(db, project_id, task_id)

    assert_is_valid_flightplan(result["kmz_bytes"])


@pytest.mark.asyncio
async def test_front_overlap_gap_logic(
    db, load_freetown_into_db, freetown_dataset_loader
):
    """Verify that a small front-overlap gap in Freetown dataset returns a valid KMZ binary file."""
    freetown_meta = freetown_dataset_loader(apply_gaps=False)
    # Create a 15-image gap (Front overlap)
    gapped_metadata = freetown_meta[:20] + freetown_meta[35:]

    project_id, batch_id, task_id = await load_freetown_into_db(
        manual_metadata=gapped_metadata
    )
    result = await identify_flight_gaps(db, project_id, task_id)

    assert len(result["kmz_bytes"]) > 100
    assert_is_valid_flightplan(result["kmz_bytes"])


@pytest.mark.asyncio
async def test_sparse_two_image_task_detects_uncovered_gap(
    db, load_freetown_into_db, freetown_dataset_loader
):
    """Very small image sets should still detect major uncovered task area."""
    freetown_meta = freetown_dataset_loader(apply_gaps=False)
    sparse_metadata = freetown_meta[:2]

    project_id, batch_id, task_id = await load_freetown_into_db(
        manual_metadata=sparse_metadata
    )
    result = await identify_flight_gaps(db, project_id, task_id)

    assert batch_id is not None
    assert result["message"] == "Successfully identified sparse-coverage gaps."
    assert result["gap_polygons"]["type"] == "FeatureCollection"
    assert result["gap_polygons"]["features"]
    assert_is_valid_flightplan(result["kmz_bytes"])


@pytest.mark.asyncio
async def test_side_overlap_gap_logic(
    db, load_freetown_into_db, freetown_dataset_loader
):
    """Verify missing side-overlap gap in Freetown dataset returns a valid KMZ binary file."""
    freetown_meta = freetown_dataset_loader(apply_gaps=False)
    # Create a large gap laterally (Side overlap)
    gapped_metadata = freetown_meta[:50] + freetown_meta[110:]

    project_id, batch_id, task_id = await load_freetown_into_db(
        manual_metadata=gapped_metadata
    )
    result = await identify_flight_gaps(db, project_id, task_id)

    assert_is_valid_flightplan(result["kmz_bytes"])


@pytest.mark.asyncio
async def test_gap_outside_aoi_ignored(db, load_freetown_into_db):
    """
    Ensures gaps found in transit (outside the Task AOI) are ignored.
    """
    # Load data with gaps
    project_id, batch_id, task_id = await load_freetown_into_db(apply_gaps=True)

    # Move AOI far from Freetown
    async with db.cursor() as cur:
        await cur.execute(
            """
            UPDATE tasks
            SET outline = ST_SetSRID(ST_MakeEnvelope(0, 0, 0.01, 0.01), 4326)
            WHERE id = %s
            """,
            (task_id,),
        )
    await db.commit()

    result = await identify_flight_gaps(db, project_id, task_id)

    assert result["kmz_bytes"] is None
    assert result["message"] == "No gaps detected"

    assert result["gap_polygons"]["type"] == "FeatureCollection"


@pytest.mark.asyncio
async def test_freetown_dataset_gap_full_verification(db, load_freetown_into_db):
    """Tests the official 'images-to-delete-for-gaps.txt' gap."""
    project_id, batch_id, task_id = await load_freetown_into_db(apply_gaps=True)
    result = await identify_flight_gaps(db, project_id, task_id)

    assert_is_valid_flightplan(result["kmz_bytes"])
    assert "Successfully identified" in result["message"]


@pytest.mark.asyncio
async def test_project_level_gap_detection_without_batch_filter(
    db, load_freetown_into_db
):
    """Project-level gap analysis should work across uploaded imagery for a task."""
    project_id, batch_id, task_id = await load_freetown_into_db(apply_gaps=True)

    result = await identify_flight_gaps(db, project_id, task_id)

    assert batch_id is not None
    assert "images" in result
    assert_is_valid_flightplan(result["kmz_bytes"])


@pytest.mark.asyncio
async def test_gap_detection_falls_back_to_exif_drone_model(db, load_freetown_into_db):
    project_id, batch_id, task_id = await load_freetown_into_db(apply_gaps=True)

    async with db.cursor() as cur:
        await cur.execute("DELETE FROM drone_flights WHERE task_id = %s", (task_id,))
        await cur.execute(
            """
            UPDATE project_images
            SET exif = exif || '{"Model":"DJI Air 3"}'::jsonb
            WHERE project_id = %s AND task_id = %s
            """,
            (project_id, task_id),
        )
    await db.commit()

    result = await identify_flight_gaps(db, project_id, task_id)

    assert result["drone_type"] == DroneType.DJI_AIR_3
    assert_is_valid_flightplan(result["kmz_bytes"])


@pytest.mark.asyncio
async def test_gap_detection_maps_fc9313_camera_alias(db, load_freetown_into_db):
    # Load a task with real-ish gap data so flight-gap generation has something to do.
    project_id, batch_id, task_id = await load_freetown_into_db(apply_gaps=True)

    async with db.cursor() as cur:
        # Remove the selected drone from the DB so the code must fall back to EXIF.
        await cur.execute("DELETE FROM drone_flights WHERE task_id = %s", (task_id,))
        await cur.execute(
            """
            UPDATE project_images
            SET exif = exif || '{"Model":"FC9313"}'::jsonb
            WHERE project_id = %s AND task_id = %s
            """,
            (project_id, task_id),
        )
        # Pretend every uploaded image came from a DJI camera that reports FC9313.
    await db.commit()

    result = await identify_flight_gaps(db, project_id, task_id)
    # If the alias works, FC9313 becomes the supported DJI Mini 5 Pro enum.

    assert batch_id is not None
    assert result["drone_type"] == DroneType.DJI_MINI_5_PRO
    assert_is_valid_flightplan(result["kmz_bytes"])


@pytest.mark.asyncio
async def test_gap_detection_without_drone_metadata_returns_clean_response(
    db, load_freetown_into_db
):
    project_id, batch_id, task_id = await load_freetown_into_db(apply_gaps=False)

    async with db.cursor() as cur:
        await cur.execute("DELETE FROM drone_flights WHERE task_id = %s", (task_id,))
    await db.commit()

    result = await identify_flight_gaps(db, project_id, task_id)

    assert batch_id is not None
    assert result["drone_type"] is None
    assert result["kmz_bytes"] is None
    assert result["message"] == "Select a drone model to generate a reflight plan."


@pytest.mark.asyncio
async def test_gap_detection_accepts_explicit_drone_override(db, load_freetown_into_db):
    project_id, batch_id, task_id = await load_freetown_into_db(apply_gaps=True)

    async with db.cursor() as cur:
        await cur.execute("DELETE FROM drone_flights WHERE task_id = %s", (task_id,))
    await db.commit()

    result = await identify_flight_gaps(
        db,
        project_id,
        task_id,
        drone_type_override=DroneType.DJI_AIR_3,
    )

    assert batch_id is not None
    assert result["drone_type"] == DroneType.DJI_AIR_3
    assert_is_valid_flightplan(result["kmz_bytes"])
