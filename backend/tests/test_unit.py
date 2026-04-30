"""
Unit Tests for Delhi Accident Severity Predictor
=================================================

Comprehensive unit tests covering each component individually:

  A) Config – paths, risk categories, risk colors, constants
  B) Delhi Data Mapper – geocode_location, haversine_distance,
     _safe_int, _create_virtual_segment, _load_generic_csv,
     DELHI_KNOWN_LOCATIONS
  C) Segment Risk Calculator – calculate_historical_risk,
     calculate_composite_risk, risk categories, risk colors
  D) Delhi Trainer – _extract_features, _assign_risk_class,
     _apply_smote
  E) Digital Twin – initialization, _get_color_scale

Usage:
  cd backend && python3 -m pytest tests/test_unit.py -v
  cd backend && python3 -m pytest tests/test_unit.py -v -m slow   # slow tests only
  cd backend && python3 -m pytest tests/test_unit.py -v -m "not slow"  # skip slow tests
"""

import os
import sys
import math
import tempfile
import pytest
import json
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, mock_open

# ---------------------------------------------------------------------------
# Ensure the backend directory is on sys.path so that `config` and `ml.*`
# can be imported regardless of the working directory.
# ---------------------------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


# ═════════════════════════════════════════════
# SHARED FIXTURES
# ═════════════════════════════════════════════

@pytest.fixture
def delhi_known_locations():
    """Return a subset of well-known Delhi locations for geocoding tests."""
    return {
        "connaught place": (28.6315, 77.2167),
        "india gate": (28.6129, 77.2295),
        "aiims": (28.5674, 77.2074),
        "kashmiri gate": (28.6676, 77.2284),
        "mukarba chowk": (28.7325, 77.1855),
    }


@pytest.fixture
def sample_segment_data_zero():
    """Segment data with zero accidents."""
    return {
        "total_accidents": 0,
        "length_m": 150,
        "severity_distribution": {
            "Fatal": 0, "Grievous": 0, "Minor": 0, "No Injury": 0
        },
        "fatal_rate": 0,
    }


@pytest.fixture
def sample_segment_data_high_fatal():
    """Segment data with high fatal accident rate."""
    return {
        "total_accidents": 10,
        "length_m": 200,
        "severity_distribution": {
            "Fatal": 8, "Grievous": 1, "Minor": 1, "No Injury": 0
        },
        "year_distribution": {"2020": 5, "2021": 5},
        "fatal_rate": 0.8,
    }


@pytest.fixture
def sample_segment_data_moderate():
    """Segment data with moderate accidents."""
    return {
        "total_accidents": 5,
        "length_m": 300,
        "severity_distribution": {
            "Fatal": 1, "Grievous": 2, "Minor": 2, "No Injury": 0
        },
        "year_distribution": {"2020": 3, "2021": 2},
        "fatal_rate": 0.2,
    }


@pytest.fixture
def sample_segment_mapping():
    """Create a minimal segment mapping dict for risk calculation tests."""
    return {
        "seg_001": {
            "segment_id": "seg_001",
            "road_name": "Test Road A",
            "road_type": "primary",
            "length_m": 200,
            "centroid_lat": 28.6139,
            "centroid_lon": 77.2090,
            "total_accidents": 10,
            "severity_distribution": {"Fatal": 5, "Grievous": 3, "Minor": 2, "No Injury": 0},
            "year_distribution": {"2020": 5, "2021": 5},
            "fatal_rate": 0.5,
        },
        "seg_002": {
            "segment_id": "seg_002",
            "road_name": "Test Road B",
            "road_type": "secondary",
            "length_m": 500,
            "centroid_lat": 28.5500,
            "centroid_lon": 77.2500,
            "total_accidents": 0,
            "severity_distribution": {"Fatal": 0, "Grievous": 0, "Minor": 0, "No Injury": 0},
            "fatal_rate": 0,
        },
    }


@pytest.fixture
def dataset2_row():
    """Sample row data mimicking delhiaccidentdataset2.csv."""
    return {
        "latitude": 28.6300,
        "longitude": 77.2300,
        "accident_severity": "Fatal",
        "road_type": "Highway",
        "date": "2023-05-15",
        "hour": 14,
        "weather": "Clear",
        "is_weekend": 0,
    }


@pytest.fixture
def dataset2_row_outside_delhi():
    """Sample row with GPS coordinates outside Delhi bounds."""
    return {
        "latitude": 19.0760,
        "longitude": 72.8777,  # Mumbai
        "accident_severity": "Minor",
        "road_type": "City Road",
        "date": "2023-03-20",
        "hour": 10,
        "weather": "Rain",
        "is_weekend": 1,
    }


@pytest.fixture
def mock_mapper():
    """Create a DelhiDataMapper with a mocked edges_gdf for unit testing."""
    import geopandas as gpd
    from shapely.geometry import LineString
    from ml.delhi_data_mapper import DelhiDataMapper

    # Create minimal GeoDataFrame
    geom = LineString([(77.20, 28.61), (77.21, 28.62)])
    edges_gdf = gpd.GeoDataFrame(
        {"name": ["Test Road"], "highway": ["primary"], "length": [100.0]},
        geometry=[geom],
        crs="EPSG:4326",
    )
    with patch("ml.delhi_data_mapper.DelhiDataMapper._prepare_edges"):
        mapper = DelhiDataMapper.__new__(DelhiDataMapper)
        mapper.edges_gdf = edges_gdf
        mapper.city_key = "delhi"
        mapper.city_config = {"center": [28.6139, 77.2090]}
        mapper.output_dir = "/tmp/test_mapped_accidents/delhi"
        mapper.mapping_path = os.path.join(mapper.output_dir, "segment_mapping.json")
        mapper.stats_path = os.path.join(mapper.output_dir, "mapping_stats.json")
        mapper.segment_mapping = {}
        mapper.all_accidents = []
        mapper.mapping_stats = {}
        # Manually initialize edge attributes
        centroids = edges_gdf.geometry.centroid
        mapper.edge_centroids_lat = centroids.y.values
        mapper.edge_centroids_lon = centroids.x.values
        edges_gdf["segment_id"] = ["seg_0"]
        mapper.road_name_index = {"test road": 0}
    return mapper


# ═════════════════════════════════════════════
# A) TEST CONFIG
# ═════════════════════════════════════════════

class TestConfig:
    """Tests for the configuration module."""

    # --- Path Constants ---

    def test_all_required_paths_are_set(self):
        """All required path constants should be defined and non-empty strings."""
        from config import (
            BASE_DIR, DATA_DIR, OUTPUTS_DIR, MODELS_DIR,
            PLOTS_DIR, SHAP_DIR, ROAD_NETWORKS_DIR,
            MAPPED_ACCIDENTS_DIR, DIGITAL_TWIN_DIR,
            DELHI_DATASETS_DIR,
        )
        for name, value in [
            ("BASE_DIR", BASE_DIR),
            ("DATA_DIR", DATA_DIR),
            ("OUTPUTS_DIR", OUTPUTS_DIR),
            ("MODELS_DIR", MODELS_DIR),
            ("PLOTS_DIR", PLOTS_DIR),
            ("SHAP_DIR", SHAP_DIR),
            ("ROAD_NETWORKS_DIR", ROAD_NETWORKS_DIR),
            ("MAPPED_ACCIDENTS_DIR", MAPPED_ACCIDENTS_DIR),
            ("DIGITAL_TWIN_DIR", DIGITAL_TWIN_DIR),
            ("DELHI_DATASETS_DIR", DELHI_DATASETS_DIR),
        ]:
            assert value, f"Path {name} is empty or None"
            assert isinstance(value, str), f"Path {name} is not a string"

    def test_delhi_datasets_dir_exists(self):
        """DELHI_DATASETS_DIR should exist on disk."""
        from config import DELHI_DATASETS_DIR
        assert os.path.isdir(DELHI_DATASETS_DIR), (
            f"DELHI_DATASETS_DIR {DELHI_DATASETS_DIR} does not exist on disk"
        )

    def test_data_dir_exists_on_disk(self):
        """DATA_DIR should exist on disk."""
        from config import DATA_DIR
        assert os.path.isdir(DATA_DIR), f"DATA_DIR {DATA_DIR} does not exist"

    # --- RISK_COLORS ---

    def test_risk_colors_has_five_entries(self):
        """RISK_COLORS should have exactly 5 entries for 5 risk tiers."""
        from config import RISK_COLORS
        assert len(RISK_COLORS) == 5, (
            f"RISK_COLORS has {len(RISK_COLORS)} entries, expected 5"
        )

    def test_risk_colors_correct_hex_values(self):
        """RISK_COLORS should contain the correct hex color for each category."""
        from config import RISK_COLORS
        expected = {
            "No Risk": "#22C55E",    # Green
            "Low": "#3B82F6",        # Blue
            "Moderate": "#EAB308",   # Yellow
            "High": "#F97316",       # Orange
            "Very High": "#EF4444",  # Red
        }
        for category, color in expected.items():
            assert category in RISK_COLORS, f"Missing category: {category}"
            assert RISK_COLORS[category] == color, (
                f"{category}: expected {color}, got {RISK_COLORS[category]}"
            )

    def test_colors_are_valid_hex(self):
        """All risk colors should be valid hex color codes (#RRGGBB)."""
        from config import RISK_COLORS
        for category, color in RISK_COLORS.items():
            assert color.startswith("#"), f"{category} color doesn't start with #"
            assert len(color) == 7, f"{category} color {color} is not 7 chars"
            # Verify all characters after # are valid hex digits
            int(color[1:], 16)  # Will raise ValueError if not valid hex

    # --- RISK_CATEGORIES ---

    def test_risk_categories_has_five_entries(self):
        """RISK_CATEGORIES should have exactly 5 categories."""
        from config import RISK_CATEGORIES
        expected = {"No Risk", "Low", "Moderate", "High", "Very High"}
        assert set(RISK_CATEGORIES.keys()) == expected

    def test_risk_categories_ranges_are_correct(self):
        """Each risk category range should match the 5-tier scheme."""
        from config import RISK_CATEGORIES
        expected_ranges = {
            "No Risk": (0, 10),
            "Low": (10, 40),
            "Moderate": (40, 60),
            "High": (60, 80),
            "Very High": (80, 101),
        }
        for category, (low, high) in expected_ranges.items():
            assert category in RISK_CATEGORIES, f"Missing category: {category}"
            assert RISK_CATEGORIES[category] == (low, high), (
                f"{category}: expected ({low}, {high}), got {RISK_CATEGORIES[category]}"
            )

    def test_risk_categories_ranges_are_contiguous(self):
        """Risk category ranges should be contiguous with no gaps."""
        from config import RISK_CATEGORIES
        sorted_cats = sorted(RISK_CATEGORIES.items(), key=lambda x: x[1][0])
        for i in range(len(sorted_cats) - 1):
            _, (_, high_current) = sorted_cats[i]
            _, (low_next, _) = sorted_cats[i + 1]
            assert high_current == low_next, (
                f"Gap between {sorted_cats[i][0]} (upper={high_current}) "
                f"and {sorted_cats[i + 1][0]} (lower={low_next})"
            )

    def test_risk_categories_start_at_zero(self):
        """First risk category should start at 0."""
        from config import RISK_CATEGORIES
        min_low = min(low for low, high in RISK_CATEGORIES.values())
        assert min_low == 0, f"Risk categories don't start at 0, start at {min_low}"

    # --- MAX_SNAP_DISTANCE_METERS ---

    def test_max_snap_distance_is_5000(self):
        """MAX_SNAP_DISTANCE_METERS should be 5000 (5 km)."""
        from config import MAX_SNAP_DISTANCE_METERS
        assert MAX_SNAP_DISTANCE_METERS == 5000

    # --- CITIES_CONFIG ---

    def test_cities_config_has_delhi(self):
        """CITIES_CONFIG should include 'delhi' key."""
        from config import CITIES_CONFIG
        assert "delhi" in CITIES_CONFIG

    def test_delhi_config_has_required_fields(self):
        """Delhi city config should have all required fields."""
        from config import CITIES_CONFIG
        delhi = CITIES_CONFIG["delhi"]
        required_fields = ["name", "display_name", "osm_query", "bbox",
                           "center", "zoom_level", "network_type"]
        for field in required_fields:
            assert field in delhi, f"Missing field '{field}' in delhi config"

    def test_delhi_bbox_reasonable(self):
        """Delhi bounding box should cover the expected Delhi area."""
        from config import CITIES_CONFIG
        bbox = CITIES_CONFIG["delhi"]["bbox"]
        assert 28.0 < bbox["lat_min"] < 29.0
        assert 28.0 < bbox["lat_max"] < 29.0
        assert 76.0 < bbox["lon_min"] < 78.0
        assert 76.0 < bbox["lon_max"] < 78.0
        assert bbox["lat_min"] < bbox["lat_max"]
        assert bbox["lon_min"] < bbox["lon_max"]

    # --- MODEL_NAMES ---

    def test_model_names_defined(self):
        """MODEL_NAMES should have the expected model list."""
        from config import MODEL_NAMES
        expected = {"RandomForest", "XGBoost", "GradientBoosting",
                    "SVM", "LogisticRegression"}
        assert set(MODEL_NAMES) == expected

    # --- API Settings ---

    def test_api_settings(self):
        """API settings should be reasonable."""
        from config import API_HOST, API_PORT
        assert API_HOST is not None
        assert isinstance(API_PORT, int)
        assert 1 <= API_PORT <= 65535


# ═════════════════════════════════════════════
# B) TEST DELHI DATA MAPPER
# ═════════════════════════════════════════════

class TestDelhiKnownLocations:
    """Tests for DELHI_KNOWN_LOCATIONS database."""

    def test_has_100_plus_entries(self):
        """DELHI_KNOWN_LOCATIONS should have 100+ entries for Delhi coverage."""
        from ml.delhi_data_mapper import DELHI_KNOWN_LOCATIONS
        assert len(DELHI_KNOWN_LOCATIONS) >= 100, (
            f"Only {len(DELHI_KNOWN_LOCATIONS)} entries, expected 100+"
        )

    def test_all_coordinates_are_in_delhi_range(self):
        """All coordinates should be within reasonable Delhi bounds."""
        from ml.delhi_data_mapper import DELHI_KNOWN_LOCATIONS
        for name, (lat, lon) in DELHI_KNOWN_LOCATIONS.items():
            assert 28.3 <= lat <= 28.95, f"{name}: lat {lat} outside Delhi"
            assert 76.7 <= lon <= 77.4, f"{name}: lon {lon} outside Delhi"

    def test_all_names_are_lowercase(self):
        """All location keys should be lowercase for consistent lookup."""
        from ml.delhi_data_mapper import DELHI_KNOWN_LOCATIONS
        for name in DELHI_KNOWN_LOCATIONS:
            assert name == name.lower(), f"Key '{name}' is not lowercase"


class TestGeocodeLocation:
    """Tests for the geocode_location() function."""

    def test_known_location_connaught_place(self):
        """Geocode 'Connaught Place' should return correct lat/lon."""
        from ml.delhi_data_mapper import geocode_location
        lat, lon = geocode_location("Connaught Place")
        assert lat is not None
        assert lon is not None
        assert abs(lat - 28.6315) < 0.01, f"Lat {lat} too far from expected"
        assert abs(lon - 77.2167) < 0.01, f"Lon {lon} too far from expected"

    def test_known_location_india_gate(self):
        """Geocode 'India Gate' should return correct lat/lon."""
        from ml.delhi_data_mapper import geocode_location
        lat, lon = geocode_location("India Gate")
        assert lat is not None
        assert lon is not None
        assert abs(lat - 28.6129) < 0.01
        assert abs(lon - 77.2295) < 0.01

    def test_known_location_aiims(self):
        """Geocode 'AIIMS' should return correct lat/lon."""
        from ml.delhi_data_mapper import geocode_location
        lat, lon = geocode_location("AIIMS")
        assert lat is not None
        assert lon is not None
        assert abs(lat - 28.5674) < 0.01
        assert abs(lon - 77.2074) < 0.01

    def test_known_location_mukarba_chowk(self):
        """Geocode 'Mukarba Chowk' should return correct lat/lon."""
        from ml.delhi_data_mapper import geocode_location
        lat, lon = geocode_location("Mukarba Chowk")
        assert lat is not None
        assert lon is not None
        assert abs(lat - 28.7325) < 0.01
        assert abs(lon - 77.1855) < 0.01

    def test_case_insensitive_lookup(self):
        """Geocode should be case-insensitive."""
        from ml.delhi_data_mapper import geocode_location
        lat1, lon1 = geocode_location("CONNAUGHT PLACE")
        lat2, lon2 = geocode_location("connaught place")
        assert lat1 == lat2
        assert lon1 == lon2

    def test_partial_match(self):
        """Geocode should work with partial name matches (e.g. 'Kashmiri Gate ISBT')."""
        from ml.delhi_data_mapper import geocode_location
        lat, lon = geocode_location("Kashmiri Gate ISBT")
        assert lat is not None
        assert lon is not None
        # Should match "kashmiri gate" or "isbt kashmiri gate"
        assert 28.3 <= lat <= 28.95
        assert 76.7 <= lon <= 77.4

    def test_unknown_location_returns_none(self):
        """Geocode with unknown/non-Delhi location should return (None, None)."""
        from ml.delhi_data_mapper import geocode_location
        lat, lon = geocode_location("xyzabc12345nonexistent")
        assert lat is None
        assert lon is None

    def test_empty_string_returns_none(self):
        """Geocode with empty string should return (None, None)."""
        from ml.delhi_data_mapper import geocode_location
        lat, lon = geocode_location("")
        assert lat is None
        assert lon is None

    def test_none_input_returns_none(self):
        """Geocode with None input should return (None, None)."""
        from ml.delhi_data_mapper import geocode_location
        lat, lon = geocode_location(None)
        assert lat is None
        assert lon is None

    def test_numeric_input_returns_none(self):
        """Geocode with non-string input should return (None, None)."""
        from ml.delhi_data_mapper import geocode_location
        lat, lon = geocode_location(12345)
        assert lat is None
        assert lon is None

    def test_whitespace_only_returns_none(self):
        """Geocode with whitespace-only string should return (None, None)."""
        from ml.delhi_data_mapper import geocode_location
        lat, lon = geocode_location("   ")
        assert lat is None
        assert lon is None

    def test_district_match(self):
        """Geocode with district names should return approximate coordinates."""
        from ml.delhi_data_mapper import geocode_location
        lat, lon = geocode_location("North District Delhi")
        assert lat is not None
        assert lon is not None
        assert 28.3 <= lat <= 28.95
        assert 76.7 <= lon <= 77.4


class TestHaversineDistance:
    """Tests for the haversine_distance() function."""

    def test_same_point_returns_zero(self):
        """Distance between the same point should be zero."""
        from ml.delhi_data_mapper import haversine_distance
        dist = haversine_distance(28.6139, 77.2090, 28.6139, 77.2090)
        assert dist == pytest.approx(0, abs=1)

    def test_known_distance_connaught_place_to_india_gate(self):
        """Distance between Connaught Place and India Gate should be ~2.5 km."""
        from ml.delhi_data_mapper import haversine_distance
        dist = haversine_distance(28.6315, 77.2167, 28.6129, 77.2295)
        assert 2000 < dist < 3000, f"Distance {dist}m outside expected range"

    def test_known_distance_aiims_to_saket(self):
        """Distance between AIIMS and Saket should be ~5 km."""
        from ml.delhi_data_mapper import haversine_distance
        dist = haversine_distance(28.5674, 77.2074, 28.5244, 77.2066)
        assert 4500 < dist < 5500, f"Distance {dist}m outside expected range"

    def test_large_distance(self):
        """Distance between Delhi and a far point should be large (~1150 km)."""
        from ml.delhi_data_mapper import haversine_distance
        dist = haversine_distance(28.6139, 77.2090, 19.0760, 72.8777)
        assert 1100000 < dist < 1200000, f"Distance {dist}m outside expected range"

    def test_symmetry(self):
        """Haversine distance should be symmetric (A->B == B->A)."""
        from ml.delhi_data_mapper import haversine_distance
        lat1, lon1 = 28.6139, 77.2090
        lat2, lon2 = 28.5500, 77.2500
        dist_ab = haversine_distance(lat1, lon1, lat2, lon2)
        dist_ba = haversine_distance(lat2, lon2, lat1, lon1)
        assert dist_ab == pytest.approx(dist_ba, rel=1e-9)

    def test_unit_is_meters(self):
        """Haversine distance should return meters, not kilometers."""
        from ml.delhi_data_mapper import haversine_distance
        # About 1 degree of latitude ≈ 111 km
        dist = haversine_distance(28.0, 77.0, 29.0, 77.0)
        assert 110000 < dist < 112000, f"Expected ~111km, got {dist}m"


class TestSafeInt:
    """Tests for DelhiDataMapper._safe_int() static method."""

    def test_normal_int(self):
        """Normal integer should be returned as-is."""
        from ml.delhi_data_mapper import DelhiDataMapper
        assert DelhiDataMapper._safe_int(42) == 42

    def test_float_converted_to_int(self):
        """Float value should be truncated to int."""
        from ml.delhi_data_mapper import DelhiDataMapper
        assert DelhiDataMapper._safe_int(3.7) == 3

    def test_string_number(self):
        """String representation of a number should be converted."""
        from ml.delhi_data_mapper import DelhiDataMapper
        assert DelhiDataMapper._safe_int("15") == 15

    def test_dash_returns_default(self):
        """Dash '-' should return the default value."""
        from ml.delhi_data_mapper import DelhiDataMapper
        assert DelhiDataMapper._safe_int("-") == 0
        assert DelhiDataMapper._safe_int("-", default=-1) == -1

    def test_nan_returns_default(self):
        """NaN value should return the default."""
        from ml.delhi_data_mapper import DelhiDataMapper
        assert DelhiDataMapper._safe_int(float("nan")) == 0
        assert DelhiDataMapper._safe_int(float("nan"), default=-1) == -1

    def test_none_returns_default(self):
        """None should return the default value."""
        from ml.delhi_data_mapper import DelhiDataMapper
        assert DelhiDataMapper._safe_int(None) == 0
        assert DelhiDataMapper._safe_int(None, default=99) == 99

    def test_empty_string_returns_default(self):
        """Empty string should return the default value."""
        from ml.delhi_data_mapper import DelhiDataMapper
        assert DelhiDataMapper._safe_int("") == 0

    def test_non_numeric_string_returns_default(self):
        """Non-numeric string should return the default value."""
        from ml.delhi_data_mapper import DelhiDataMapper
        assert DelhiDataMapper._safe_int("abc") == 0

    def test_negative_number(self):
        """Negative number should be handled correctly."""
        from ml.delhi_data_mapper import DelhiDataMapper
        assert DelhiDataMapper._safe_int(-5) == -5


class TestCreateVirtualSegment:
    """Tests for _create_virtual_segment() method."""

    def test_creates_correct_segment_structure(self):
        """Virtual segment should have all required fields."""
        from ml.delhi_data_mapper import DelhiDataMapper
        mapper = MagicMock(spec=DelhiDataMapper)
        # Call the real method
        result = DelhiDataMapper._create_virtual_segment(
            mapper, 28.6139, 77.2090, "Test Road", {}
        )
        required_fields = [
            "segment_id", "road_name", "road_type", "length_m",
            "centroid_lat", "centroid_lon", "is_virtual",
            "total_accidents", "severity_distribution",
            "time_distribution", "weather_distribution",
            "year_distribution", "fatal_rate", "accidents",
            "source_datasets",
        ]
        for field in required_fields:
            assert field in result, f"Missing field '{field}' in virtual segment"

    def test_virtual_segment_is_flagged(self):
        """Virtual segment should have is_virtual=True."""
        from ml.delhi_data_mapper import DelhiDataMapper
        mapper = MagicMock(spec=DelhiDataMapper)
        result = DelhiDataMapper._create_virtual_segment(
            mapper, 28.6139, 77.2090, "Test Road", {}
        )
        assert result["is_virtual"] is True

    def test_virtual_segment_coordinates(self):
        """Virtual segment should store the provided GPS coordinates."""
        from ml.delhi_data_mapper import DelhiDataMapper
        mapper = MagicMock(spec=DelhiDataMapper)
        result = DelhiDataMapper._create_virtual_segment(
            mapper, 28.5000, 77.1000, "Some Road", {}
        )
        assert result["centroid_lat"] == 28.5
        assert result["centroid_lon"] == 77.1

    def test_virtual_segment_road_name(self):
        """Virtual segment should use provided road name or default."""
        from ml.delhi_data_mapper import DelhiDataMapper
        mapper = MagicMock(spec=DelhiDataMapper)
        result = DelhiDataMapper._create_virtual_segment(
            mapper, 28.5, 77.1, "My Road", {}
        )
        assert result["road_name"] == "My Road"

        result2 = DelhiDataMapper._create_virtual_segment(
            mapper, 28.5, 77.1, None, {}
        )
        assert result2["road_name"] == "Virtual Segment"

    def test_virtual_segment_zero_accidents(self):
        """Virtual segment should start with zero accidents."""
        from ml.delhi_data_mapper import DelhiDataMapper
        mapper = MagicMock(spec=DelhiDataMapper)
        result = DelhiDataMapper._create_virtual_segment(
            mapper, 28.5, 77.1, "Road", {}
        )
        assert result["total_accidents"] == 0
        for count in result["severity_distribution"].values():
            assert count == 0

    def test_virtual_segment_road_type_from_record(self):
        """Virtual segment should extract road_type from the record if available."""
        from ml.delhi_data_mapper import DelhiDataMapper
        mapper = MagicMock(spec=DelhiDataMapper)
        result = DelhiDataMapper._create_virtual_segment(
            mapper, 28.5, 77.1, "Road", {"road_type": "Highway"}
        )
        assert result["road_type"] == "Highway"

    def test_virtual_segment_unknown_road_type_when_missing(self):
        """Virtual segment should default road_type to 'unknown' when not in record."""
        from ml.delhi_data_mapper import DelhiDataMapper
        mapper = MagicMock(spec=DelhiDataMapper)
        result = DelhiDataMapper._create_virtual_segment(
            mapper, 28.5, 77.1, "Road", {}
        )
        assert result["road_type"] == "unknown"


class TestLoadGenericCsv:
    """Tests for _load_generic_csv() method."""

    def test_load_sample_csv(self, mock_mapper):
        """_load_generic_csv should load records from a valid CSV file."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                         delete=False) as f:
            f.write("Zone_Name,Fatal,Simple,Total\n")
            f.write("Kashmiri Gate,5,10,15\n")
            f.write("Connaught Place,2,8,10\n")
            tmp_path = f.name

        try:
            records = mock_mapper._load_generic_csv(
                tmp_path, source_name="test_csv", year="2023",
            )
            assert isinstance(records, list)
            assert len(records) == 2
            for rec in records:
                assert rec["source"] == "test_csv"
                assert rec["year"] == "2023"
                assert "location_name" in rec
                assert "total_accidents" in rec
        finally:
            os.unlink(tmp_path)

    def test_load_nonexistent_csv_returns_empty(self, mock_mapper):
        """_load_generic_csv with non-existent path should return empty list."""
        records = mock_mapper._load_generic_csv(
            "/nonexistent/path/file.csv",
            source_name="missing", year="2023",
        )
        assert records == []

    def test_load_empty_csv_returns_empty(self, mock_mapper):
        """_load_generic_csv with an empty CSV should return empty list."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                         delete=False) as f:
            f.write("Zone_Name,Fatal,Simple,Total\n")  # header only
            tmp_path = f.name

        try:
            records = mock_mapper._load_generic_csv(
                tmp_path, source_name="empty_csv", year="2023",
            )
            assert records == []
        finally:
            os.unlink(tmp_path)

    def test_load_csv_with_custom_columns(self, mock_mapper):
        """_load_generic_csv should respect custom column options."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                         delete=False) as f:
            f.write("Road_Name,Killed,Injured,Total_Crashes\n")
            f.write("Ring Road,3,7,10\n")
            tmp_path = f.name

        try:
            records = mock_mapper._load_generic_csv(
                tmp_path, source_name="custom_cols", year="2023",
                location_col_options=["Road_Name"],
                crash_col_options={
                    "fatal": ["Killed"],
                    "simple": ["Injured"],
                    "total": ["Total_Crashes"],
                },
            )
            assert len(records) == 1
            assert records[0]["fatal_accidents"] == 3
            assert records[0]["grievous_accidents"] == 7
            assert records[0]["total_accidents"] == 10
        finally:
            os.unlink(tmp_path)


class TestParseDataset2Row:
    """Tests for the _parse_dataset2_row() method of DelhiDataMapper."""

    def test_parse_valid_fatal_row(self, mock_mapper, dataset2_row):
        """Parsing a valid fatal accident row should return correct data."""
        result = mock_mapper._parse_dataset2_row(dataset2_row)
        assert result is not None
        assert result["severity"] == "Fatal"
        assert result["fatal_accidents"] == 1
        assert result["grievous_accidents"] == 0
        assert result["minor_accidents"] == 0
        assert result["total_accidents"] == 1
        assert abs(result["latitude"] - 28.6300) < 0.001
        assert abs(result["longitude"] - 77.2300) < 0.001

    def test_parse_severe_accident(self, mock_mapper):
        """Parsing a severe accident row should return 'Fatal' severity."""
        row = {
            "latitude": 28.6500, "longitude": 77.2300,
            "accident_severity": "Severe",
            "road_type": "Highway", "date": "2023-01-15",
            "hour": 8, "weather": "Clear", "is_weekend": 0,
        }
        result = mock_mapper._parse_dataset2_row(row)
        assert result is not None
        assert result["severity"] == "Fatal"

    def test_parse_grievous_accident(self, mock_mapper):
        """Parsing a serious/grievous accident should return 'Grievous' severity."""
        row = {
            "latitude": 28.6500, "longitude": 77.2300,
            "accident_severity": "Serious Injury",
            "road_type": "City Road", "date": "2023-06-10",
            "hour": 15, "weather": "Rain", "is_weekend": 0,
        }
        result = mock_mapper._parse_dataset2_row(row)
        assert result is not None
        assert result["severity"] == "Grievous"

    def test_parse_minor_accident(self, mock_mapper):
        """Parsing a minor accident should return 'Minor' severity."""
        row = {
            "latitude": 28.5500, "longitude": 77.2000,
            "accident_severity": "Slight Injury",
            "road_type": "Local Road", "date": "2023-09-05",
            "hour": 11, "weather": "Clear", "is_weekend": 1,
        }
        result = mock_mapper._parse_dataset2_row(row)
        assert result is not None
        assert result["severity"] == "Minor"

    def test_parse_outside_delhi_returns_none(self, mock_mapper, dataset2_row_outside_delhi):
        """Parsing a row with GPS outside Delhi bounds should return None."""
        result = mock_mapper._parse_dataset2_row(dataset2_row_outside_delhi)
        assert result is None

    def test_parse_missing_gps_returns_none(self, mock_mapper):
        """Parsing a row with missing GPS should return None."""
        row = {"latitude": None, "longitude": None, "accident_severity": "Fatal"}
        result = mock_mapper._parse_dataset2_row(row)
        assert result is None

    def test_parse_zero_gps_returns_none(self, mock_mapper):
        """Parsing a row with zero GPS should return None (outside Delhi)."""
        row = {"latitude": 0, "longitude": 0, "accident_severity": "Fatal"}
        result = mock_mapper._parse_dataset2_row(row)
        assert result is None


# ═════════════════════════════════════════════
# C) TEST SEGMENT RISK CALCULATOR
# ═════════════════════════════════════════════

class TestCalculateHistoricalRisk:
    """Tests for calculate_historical_risk() method."""

    @pytest.fixture
    def risk_calc(self, sample_segment_mapping):
        """Create a SegmentRiskCalculator instance."""
        from ml.segment_risk_calculator import SegmentRiskCalculator
        return SegmentRiskCalculator(sample_segment_mapping, city_key="delhi",
                                     predictor=None)

    def test_zero_accidents_returns_zero(self, risk_calc, sample_segment_data_zero):
        """Segment with zero accidents should have risk score 0."""
        risk = risk_calc.calculate_historical_risk(sample_segment_data_zero)
        assert risk == 0.0

    def test_high_fatal_rate_produces_high_risk(self, risk_calc, sample_segment_data_high_fatal):
        """Segment with high fatal rate should produce high risk score (>50)."""
        risk = risk_calc.calculate_historical_risk(sample_segment_data_high_fatal)
        assert risk > 50, f"Expected high risk (>50), got {risk}"

    def test_risk_score_in_range(self, risk_calc, sample_segment_data_moderate):
        """Risk score should always be in 0-100 range."""
        risk = risk_calc.calculate_historical_risk(sample_segment_data_moderate)
        assert 0 <= risk <= 100, f"Risk {risk} outside 0-100 range"

    def test_high_fatal_rate_multiplier(self, risk_calc):
        """Higher fatality rate should produce higher risk than same count of minor."""
        data_minor = {
            "total_accidents": 2, "length_m": 5000,
            "severity_distribution": {"Fatal": 0, "Grievous": 0, "Minor": 2, "No Injury": 0},
            "year_distribution": {"2020": 1, "2021": 1}, "fatal_rate": 0.0,
        }
        data_fatal = {
            "total_accidents": 2, "length_m": 5000,
            "severity_distribution": {"Fatal": 2, "Grievous": 0, "Minor": 0, "No Injury": 0},
            "year_distribution": {"2020": 1, "2021": 1}, "fatal_rate": 1.0,
        }
        risk_minor = risk_calc.calculate_historical_risk(data_minor)
        risk_fatal = risk_calc.calculate_historical_risk(data_fatal)
        assert risk_fatal > risk_minor, (
            f"Fatal risk ({risk_fatal}) should be > minor risk ({risk_minor})"
        )

    def test_no_injury_only_low_risk(self, risk_calc):
        """Segment with only 'No Injury' accidents should have low risk (<30)."""
        data = {
            "total_accidents": 5, "length_m": 500,
            "severity_distribution": {"Fatal": 0, "Grievous": 0, "Minor": 0, "No Injury": 5},
            "year_distribution": {"2020": 3, "2021": 2}, "fatal_rate": 0.0,
        }
        risk = risk_calc.calculate_historical_risk(data)
        assert risk < 30, f"Expected low risk for no-injury only, got {risk}"

    def test_short_segment_higher_risk(self, risk_calc):
        """Same accident count on a shorter segment should yield higher risk (density)."""
        data_short = {
            "total_accidents": 3, "length_m": 100,
            "severity_distribution": {"Fatal": 1, "Grievous": 1, "Minor": 1, "No Injury": 0},
            "year_distribution": {"2020": 2, "2021": 1}, "fatal_rate": 0.33,
        }
        data_long = {
            "total_accidents": 3, "length_m": 5000,
            "severity_distribution": {"Fatal": 1, "Grievous": 1, "Minor": 1, "No Injury": 0},
            "year_distribution": {"2020": 2, "2021": 1}, "fatal_rate": 0.33,
        }
        risk_short = risk_calc.calculate_historical_risk(data_short)
        risk_long = risk_calc.calculate_historical_risk(data_long)
        assert risk_short > risk_long, (
            f"Short segment risk ({risk_short}) should be > long segment ({risk_long})"
        )


class TestCalculateCompositeRisk:
    """Tests for calculate_composite_risk() method."""

    @pytest.fixture
    def risk_calc(self, sample_segment_mapping):
        """Create a SegmentRiskCalculator instance."""
        from ml.segment_risk_calculator import SegmentRiskCalculator
        return SegmentRiskCalculator(sample_segment_mapping, city_key="delhi",
                                     predictor=None)

    def test_returns_all_required_fields(self, risk_calc, sample_segment_data_moderate):
        """calculate_composite_risk() should return all required fields."""
        result = risk_calc.calculate_composite_risk(sample_segment_data_moderate)
        required_fields = [
            "historical_risk", "predictive_risk",
            "composite_risk", "risk_category", "risk_color",
        ]
        for field in required_fields:
            assert field in result, f"Missing field '{field}' in result"

    def test_composite_risk_is_weighted_average(self, risk_calc, sample_segment_data_moderate):
        """Composite risk should be weighted average of historical and predictive."""
        from config import HISTORICAL_RISK_WEIGHT, PREDICTIVE_RISK_WEIGHT
        result = risk_calc.calculate_composite_risk(sample_segment_data_moderate)
        expected = (
            HISTORICAL_RISK_WEIGHT * result["historical_risk"]
            + PREDICTIVE_RISK_WEIGHT * result["predictive_risk"]
        )
        assert abs(result["composite_risk"] - round(expected, 2)) < 0.01

    def test_composite_risk_in_range(self, risk_calc, sample_segment_data_moderate):
        """Composite risk should be in 0-100 range."""
        result = risk_calc.calculate_composite_risk(sample_segment_data_moderate)
        assert 0 <= result["composite_risk"] <= 100

    def test_predictive_risk_is_50_when_no_predictor(self, risk_calc,
                                                      sample_segment_data_moderate):
        """Without a predictor, predictive risk should default to 50."""
        result = risk_calc.calculate_composite_risk(sample_segment_data_moderate)
        assert result["predictive_risk"] == 50.0


class TestRiskCategories:
    """Tests for risk category classification matching the 5-tier scheme."""

    def _category_for_score(self, score):
        """Helper: get risk category for a given composite score."""
        from config import RISK_CATEGORIES
        for category, (low, high) in RISK_CATEGORIES.items():
            if low <= score < high:
                return category
        return "Very High"  # >= 100 edge case

    def test_no_risk_range(self):
        """Scores 0-10 should be classified as 'No Risk'."""
        for score in [0, 5, 9.99]:
            cat = self._category_for_score(score)
            assert cat == "No Risk", f"Score {score} classified as {cat}"

    def test_low_risk_range(self):
        """Scores 10-40 should be classified as 'Low'."""
        for score in [10, 25, 39.99]:
            cat = self._category_for_score(score)
            assert cat == "Low", f"Score {score} classified as {cat}"

    def test_moderate_risk_range(self):
        """Scores 40-60 should be classified as 'Moderate'."""
        for score in [40, 50, 59.99]:
            cat = self._category_for_score(score)
            assert cat == "Moderate", f"Score {score} classified as {cat}"

    def test_high_risk_range(self):
        """Scores 60-80 should be classified as 'High'."""
        for score in [60, 70, 79.99]:
            cat = self._category_for_score(score)
            assert cat == "High", f"Score {score} classified as {cat}"

    def test_very_high_risk_range(self):
        """Scores 80+ should be classified as 'Very High'."""
        for score in [80, 90, 95, 100]:
            cat = self._category_for_score(score)
            assert cat == "Very High", f"Score {score} classified as {cat}"


class TestRiskColors:
    """Tests for risk color mapping matching the config."""

    def test_no_risk_color_is_green(self):
        """'No Risk' category should map to green color."""
        from config import RISK_COLORS
        assert RISK_COLORS["No Risk"] == "#22C55E"

    def test_low_risk_color_is_blue(self):
        """'Low' category should map to blue color."""
        from config import RISK_COLORS
        assert RISK_COLORS["Low"] == "#3B82F6"

    def test_moderate_risk_color_is_yellow(self):
        """'Moderate' category should map to yellow color."""
        from config import RISK_COLORS
        assert RISK_COLORS["Moderate"] == "#EAB308"

    def test_high_risk_color_is_orange(self):
        """'High' category should map to orange color."""
        from config import RISK_COLORS
        assert RISK_COLORS["High"] == "#F97316"

    def test_very_high_risk_color_is_red(self):
        """'Very High' category should map to red color."""
        from config import RISK_COLORS
        assert RISK_COLORS["Very High"] == "#EF4444"

    def test_all_categories_have_colors(self):
        """All risk categories should have corresponding color entries."""
        from config import RISK_CATEGORIES, RISK_COLORS
        for category in RISK_CATEGORIES:
            assert category in RISK_COLORS, f"Missing color for category: {category}"


# ═════════════════════════════════════════════
# D) TEST DELHI TRAINER
# ═════════════════════════════════════════════

class TestExtractFeatures:
    """Tests for _extract_features() function in delhi_trainer."""

    def test_with_valid_segment_data(self):
        """_extract_features() with valid segment data should return all feature keys."""
        from ml.delhi_trainer import _extract_features, FEATURE_COLUMNS
        segment = {
            "total_accidents": 10,
            "severity_distribution": {"Fatal": 3, "Grievous": 4, "Minor": 3, "No Injury": 0},
            "fatal_rate": 0.3,
            "length_m": 200,
            "year_distribution": {"2020": 5, "2021": 5},
            "time_distribution": {"Morning": 2, "Afternoon": 3, "Evening": 4, "Night": 1},
            "road_type": "primary",
        }
        result = _extract_features(segment)
        assert result is not None
        for col in FEATURE_COLUMNS:
            assert col in result, f"Missing feature column: {col}"
        assert result["total_accidents"] == 10
        assert result["fatal_count"] == 3
        assert result["grievous_count"] == 4
        assert result["minor_count"] == 3

    def test_with_zero_accidents_returns_none(self):
        """_extract_features() with zero accidents should return None."""
        from ml.delhi_trainer import _extract_features
        segment = {
            "total_accidents": 0,
            "severity_distribution": {"Fatal": 0, "Grievous": 0, "Minor": 0, "No Injury": 0},
            "fatal_rate": 0.0, "length_m": 200, "road_type": "primary",
        }
        result = _extract_features(segment)
        assert result is None

    def test_with_negative_accidents_returns_none(self):
        """_extract_features() with negative total_accidents should return None."""
        from ml.delhi_trainer import _extract_features
        segment = {"total_accidents": -1, "severity_distribution": {}, "fatal_rate": 0}
        result = _extract_features(segment)
        assert result is None

    def test_accident_density_calculation(self):
        """Accident density should be total_accidents / (length_km)."""
        from ml.delhi_trainer import _extract_features
        segment = {
            "total_accidents": 20,
            "severity_distribution": {"Fatal": 5, "Grievous": 5, "Minor": 10, "No Injury": 0},
            "fatal_rate": 0.25, "length_m": 500,
            "year_distribution": {"2020": 10, "2021": 10},
            "time_distribution": {"Morning": 5, "Afternoon": 5, "Evening": 5, "Night": 5},
            "road_type": "residential",
        }
        result = _extract_features(segment)
        assert result is not None
        # density = 20 / (500/1000) = 20 / 0.5 = 40
        assert result["accident_density"] == 40.0

    def test_weighted_severity_calculation(self):
        """Weighted severity should be (Fatal*10 + Grievous*5 + Minor*2) / total."""
        from ml.delhi_trainer import _extract_features
        segment = {
            "total_accidents": 10,
            "severity_distribution": {"Fatal": 2, "Grievous": 3, "Minor": 5, "No Injury": 0},
            "fatal_rate": 0.2, "length_m": 1000,
            "year_distribution": {"2020": 10},
            "time_distribution": {"Morning": 3, "Afternoon": 3, "Evening": 2, "Night": 2},
            "road_type": "primary",
        }
        result = _extract_features(segment)
        assert result is not None
        # weighted = (2*10 + 3*5 + 5*2) / 10 = (20 + 15 + 10) / 10 = 4.5
        assert result["weighted_severity"] == 4.5

    def test_highway_flag(self):
        """Segments with trunk/motorway/primary road type should have is_highway=1."""
        from ml.delhi_trainer import _extract_features
        for road_type in ["trunk", "motorway", "primary"]:
            segment = {
                "total_accidents": 5,
                "severity_distribution": {"Fatal": 1, "Grievous": 2, "Minor": 2, "No Injury": 0},
                "fatal_rate": 0.2, "length_m": 500,
                "year_distribution": {"2020": 5},
                "time_distribution": {"Morning": 2, "Afternoon": 3},
                "road_type": road_type,
            }
            result = _extract_features(segment)
            assert result["is_highway"] == 1, f"Expected is_highway=1 for '{road_type}'"

    def test_urban_flag(self):
        """Segments with residential/tertiary/living_street type should have is_urban=1."""
        from ml.delhi_trainer import _extract_features
        for road_type in ["residential", "tertiary", "living_street"]:
            segment = {
                "total_accidents": 5,
                "severity_distribution": {"Fatal": 1, "Grievous": 2, "Minor": 2, "No Injury": 0},
                "fatal_rate": 0.2, "length_m": 500,
                "year_distribution": {"2020": 5},
                "time_distribution": {"Morning": 2, "Afternoon": 3},
                "road_type": road_type,
            }
            result = _extract_features(segment)
            assert result["is_urban"] == 1, f"Expected is_urban=1 for '{road_type}'"


class TestAssignRiskClass:
    """Tests for _assign_risk_class() function."""

    def test_high_risk_class(self):
        """High fatal_rate or high total_accidents should classify as 'High'."""
        from ml.delhi_trainer import _assign_risk_class
        # High fatal_rate
        assert _assign_risk_class({"total_accidents": 5, "fatal_rate": 0.3}) == "High"
        # High total_accidents
        assert _assign_risk_class({"total_accidents": 25, "fatal_rate": 0.05}) == "High"

    def test_medium_risk_class(self):
        """Medium fatal_rate or moderate total_accidents should classify as 'Medium'."""
        from ml.delhi_trainer import _assign_risk_class
        assert _assign_risk_class({"total_accidents": 8, "fatal_rate": 0.1}) == "Medium"
        assert _assign_risk_class({"total_accidents": 5, "fatal_rate": 0.05}) == "Medium"

    def test_low_risk_class(self):
        """Low fatal_rate and low total_accidents should classify as 'Low'."""
        from ml.delhi_trainer import _assign_risk_class
        assert _assign_risk_class({"total_accidents": 2, "fatal_rate": 0.01}) == "Low"
        assert _assign_risk_class({"total_accidents": 3, "fatal_rate": 0.0}) == "Low"

    def test_zero_accidents_returns_none(self):
        """Segment with zero accidents should return None."""
        from ml.delhi_trainer import _assign_risk_class
        assert _assign_risk_class({"total_accidents": 0, "fatal_rate": 0.0}) is None

    def test_negative_accidents_returns_none(self):
        """Segment with negative accidents should return None."""
        from ml.delhi_trainer import _assign_risk_class
        assert _assign_risk_class({"total_accidents": -5, "fatal_rate": 0.0}) is None

    def test_boundary_high_risk(self):
        """Test exact boundary values for High risk (fatal_rate=0.2, total=20)."""
        from ml.delhi_trainer import _assign_risk_class
        assert _assign_risk_class({"total_accidents": 5, "fatal_rate": 0.2}) == "High"
        assert _assign_risk_class({"total_accidents": 20, "fatal_rate": 0.01}) == "High"

    def test_boundary_medium_risk(self):
        """Test exact boundary values for Medium risk (fatal_rate=0.05, total=5)."""
        from ml.delhi_trainer import _assign_risk_class
        assert _assign_risk_class({"total_accidents": 5, "fatal_rate": 0.05}) == "Medium"


class TestApplySmote:
    """Tests for _apply_smote() function."""

    def test_balanced_data(self):
        """SMOTE on balanced data should still return valid arrays."""
        from ml.delhi_trainer import _apply_smote
        X = np.random.rand(30, 5)
        y = np.array([0] * 10 + [1] * 10 + [2] * 10)
        X_res, y_res, method = _apply_smote(X, y)
        assert X_res.shape[0] >= X.shape[0], "Resampled should be >= original"
        assert method in ["SMOTE", "SMOTETomek", "None"]

    def test_imbalanced_data(self):
        """SMOTE on imbalanced data should increase minority class samples."""
        from ml.delhi_trainer import _apply_smote
        X = np.random.rand(60, 5)
        y = np.array([0] * 50 + [1] * 10)
        X_res, y_res, method = _apply_smote(X, y)
        if method != "None":
            # Minority class should be upsampled
            unique, counts = np.unique(y_res, return_counts=True)
            assert min(counts) >= 10, "Minority class should be upsampled"
        assert X_res.shape[0] >= X.shape[0]

    def test_single_class_returns_original(self):
        """SMOTE with a single class should return original data (method='None')."""
        from ml.delhi_trainer import _apply_smote
        X = np.random.rand(20, 5)
        y = np.zeros(20)
        X_res, y_res, method = _apply_smote(X, y)
        # With only one class SMOTE can't work, should return original
        assert method == "None" or X_res.shape[0] == X.shape[0]

    def test_returns_correct_tuple_structure(self):
        """_apply_smote should return (X_res, y_res, method_name) tuple."""
        from ml.delhi_trainer import _apply_smote
        X = np.random.rand(30, 5)
        y = np.array([0] * 15 + [1] * 15)
        result = _apply_smote(X, y)
        assert len(result) == 3
        X_res, y_res, method = result
        assert isinstance(X_res, np.ndarray)
        assert isinstance(y_res, np.ndarray)
        assert isinstance(method, str)


# ═════════════════════════════════════════════
# E) TEST DIGITAL TWIN
# ═════════════════════════════════════════════

class TestDigitalTwin:
    """Tests for DigitalTwin class."""

    @pytest.fixture(autouse=True)
    def _check_osmnx(self):
        """Skip DigitalTwin tests if osmnx is not installed."""
        pytest.importorskip("osmnx", reason="osmnx not installed, skipping")

    def test_initialization_for_delhi(self):
        """DigitalTwin should initialize correctly for Delhi."""
        from ml.digital_twin import DigitalTwin
        twin = DigitalTwin("delhi")
        assert twin.city_key == "delhi"
        assert twin.city_name == "Delhi"
        assert twin.metadata["status"] == "not_initialized"
        assert twin.road_network is None
        assert twin.edges_gdf is None

    def test_initialization_for_dehradun(self):
        """DigitalTwin should initialize correctly for Dehradun."""
        from ml.digital_twin import DigitalTwin
        twin = DigitalTwin("dehradun")
        assert twin.city_key == "dehradun"
        assert twin.city_name == "Dehradun"

    def test_initialization_for_bangalore(self):
        """DigitalTwin should initialize correctly for Bangalore."""
        from ml.digital_twin import DigitalTwin
        twin = DigitalTwin("bangalore")
        assert twin.city_key == "bangalore"
        assert twin.city_name == "Bangalore"

    def test_invalid_city_key_raises_value_error(self):
        """DigitalTwin with invalid city key should raise ValueError."""
        from ml.digital_twin import DigitalTwin
        with pytest.raises(ValueError, match="not found in CITIES_CONFIG"):
            DigitalTwin("mumbai")

    def test_invalid_city_key_empty_raises_value_error(self):
        """DigitalTwin with empty city key should raise ValueError."""
        from ml.digital_twin import DigitalTwin
        with pytest.raises(ValueError):
            DigitalTwin("")

    def test_initialization_with_predictor(self):
        """DigitalTwin should accept an optional predictor parameter."""
        from ml.digital_twin import DigitalTwin
        mock_predictor = MagicMock()
        twin = DigitalTwin("delhi", predictor=mock_predictor)
        assert twin.predictor is mock_predictor

    def test_initialization_without_predictor(self):
        """DigitalTwin should work with predictor=None."""
        from ml.digital_twin import DigitalTwin
        twin = DigitalTwin("delhi", predictor=None)
        assert twin.predictor is None

    def test_get_metadata_returns_dict(self):
        """get_metadata() should return a dict with required keys."""
        from ml.digital_twin import DigitalTwin
        twin = DigitalTwin("delhi")
        metadata = twin.get_metadata()
        assert isinstance(metadata, dict)
        assert "city_key" in metadata
        assert "status" in metadata

    def test_output_dir_created(self):
        """Initialization should create the output directory."""
        from ml.digital_twin import DigitalTwin
        from config import DIGITAL_TWIN_DIR
        twin = DigitalTwin("delhi")
        expected_dir = os.path.join(DIGITAL_TWIN_DIR, "delhi")
        assert os.path.isdir(expected_dir)


class TestGetColorScale:
    """Tests for _get_color_scale() method of DigitalTwin."""

    @pytest.fixture(autouse=True)
    def _check_osmnx(self):
        """Skip if osmnx is not installed."""
        pytest.importorskip("osmnx")

    def test_returns_five_levels(self):
        """_get_color_scale() should return exactly 5 color levels."""
        from ml.digital_twin import DigitalTwin
        twin = DigitalTwin("delhi")
        scale = twin._get_color_scale()
        assert isinstance(scale, list)
        assert len(scale) == 5

    def test_scale_has_required_fields(self):
        """Each color scale entry should have category, color, range_min, range_max."""
        from ml.digital_twin import DigitalTwin
        twin = DigitalTwin("delhi")
        scale = twin._get_color_scale()
        for entry in scale:
            assert "category" in entry
            assert "color" in entry
            assert "range_min" in entry
            assert "range_max" in entry

    def test_scale_categories_in_order(self):
        """Color scale categories should be in ascending risk order."""
        from ml.digital_twin import DigitalTwin
        twin = DigitalTwin("delhi")
        scale = twin._get_color_scale()
        expected_order = ["No Risk", "Low", "Moderate", "High", "Very High"]
        actual_order = [entry["category"] for entry in scale]
        assert actual_order == expected_order

    def test_scale_colors_match_config(self):
        """Color scale colors should match RISK_COLORS config."""
        from ml.digital_twin import DigitalTwin
        from config import RISK_COLORS
        twin = DigitalTwin("delhi")
        scale = twin._get_color_scale()
        for entry in scale:
            assert entry["color"] == RISK_COLORS[entry["category"]]


# ═════════════════════════════════════════════
# F) TEST PREDICTOR (bonus unit tests)
# ═════════════════════════════════════════════

class TestAccidentPredictor:
    """Tests for AccidentPredictor class."""

    def test_initialization_primary(self):
        """Predictor should initialize correctly with 'primary' dataset key."""
        from ml.predictor import AccidentPredictor
        predictor = AccidentPredictor(dataset_key="primary")
        assert predictor.dataset_key == "primary"
        assert predictor.models == {}
        assert predictor.scaler is None
        assert predictor.label_encoder is None
        assert predictor.feature_names == []
        assert predictor.label_mapping == {}
        assert predictor.loaded is False

    def test_initialization_secondary(self):
        """Predictor should initialize correctly with 'secondary' dataset key."""
        from ml.predictor import AccidentPredictor
        predictor = AccidentPredictor(dataset_key="secondary")
        assert predictor.dataset_key == "secondary"
        assert predictor.loaded is False

    def test_get_available_models_before_loading(self):
        """get_available_models() should return empty list before loading."""
        from ml.predictor import AccidentPredictor
        predictor = AccidentPredictor()
        available = predictor.get_available_models()
        assert isinstance(available, list)
        assert len(available) == 0

    def test_get_feature_names_before_loading(self):
        """get_feature_names() should return empty list before loading."""
        from ml.predictor import AccidentPredictor
        predictor = AccidentPredictor()
        features = predictor.get_feature_names()
        assert isinstance(features, list)
        assert len(features) == 0

    def test_get_label_mapping_before_loading(self):
        """get_label_mapping() should return empty dict before loading."""
        from ml.predictor import AccidentPredictor
        predictor = AccidentPredictor()
        mapping = predictor.get_label_mapping()
        assert isinstance(mapping, dict)
        assert len(mapping) == 0

    def test_predict_before_loading_returns_error(self):
        """predict() before loading should return error dict."""
        from ml.predictor import AccidentPredictor
        predictor = AccidentPredictor()
        result = predictor.predict({"test": "input"})
        assert "error" in result
        assert ("not loaded" in result["error"].lower()
                or "no models" in result["error"].lower())


# ═════════════════════════════════════════════
# IMPORT FIXTURE FOR DELHI DATA MAPPER TESTS
# ═════════════════════════════════════════════

# We need to import DelhiDataMapper for the mock_mapper fixtures
from ml.delhi_data_mapper import DelhiDataMapper
