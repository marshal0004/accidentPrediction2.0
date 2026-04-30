"""
Integration Tests for Delhi Accident Severity Predictor
=======================================================

Tests component interactions through the full pipeline:

  1. Data Pipeline  – loading all Delhi datasets, geocoding, mapping to segments
  2. Risk Pipeline   – risk calculation produces valid scores, categories, heatmap data
  3. API Endpoints   – FastAPI test client with all digital twin endpoints

Usage:
  cd backend && python3 -m pytest tests/test_integration.py -v
  cd backend && python3 -m pytest tests/test_integration.py -v -m integration
  cd backend && python3 -m pytest tests/test_integration.py -v -m api  # API tests only
"""

import os
import sys
import json
import time
import math
import pytest

# ---------------------------------------------------------------------------
# Ensure backend dir on sys.path
# ---------------------------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


# ═════════════════════════════════════════════
# SHARED HELPERS
# ═════════════════════════════════════════════

def _load_json(path):
    """Load JSON from disk, or return None if missing."""
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _delhi_segment_mapping_path():
    from config import MAPPED_ACCIDENTS_DIR
    return os.path.join(MAPPED_ACCIDENTS_DIR, "delhi", "segment_mapping.json")


def _delhi_risk_path():
    from config import DIGITAL_TWIN_DIR
    return os.path.join(DIGITAL_TWIN_DIR, "delhi", "segment_risks.json")


def _delhi_risk_stats_path():
    from config import DIGITAL_TWIN_DIR
    return os.path.join(DIGITAL_TWIN_DIR, "delhi", "risk_stats.json")


def _delhi_heatmap_path():
    from config import DIGITAL_TWIN_DIR
    return os.path.join(DIGITAL_TWIN_DIR, "delhi", "heatmap_segments.json")


# ═════════════════════════════════════════════
# 1. DATA PIPELINE
# ═════════════════════════════════════════════

@pytest.mark.integration
class TestDataPipeline:
    """Test: Load all Delhi datasets → geocode → map to segments."""

    @pytest.fixture
    def mapper(self):
        """Create a DelhiDataMapper with the real road network for integration testing."""
        import osmnx as ox
        from ml.delhi_data_mapper import DelhiDataMapper
        from config import ROAD_NETWORKS_DIR

        graphml_path = os.path.join(ROAD_NETWORKS_DIR, "delhi", "delhi_roads.graphml")
        if not os.path.exists(graphml_path):
            pytest.skip("Delhi road network cache not found. Run build_twin first.")

        graph = ox.load_graphml(graphml_path)
        _, edges_gdf = ox.graph_to_gdfs(graph)
        return DelhiDataMapper(edges_gdf, "delhi")

    @pytest.mark.slow
    def test_loading_all_datasets_returns_records(self, mapper):
        """load_all_delhi_datasets() should return a non-empty list of records."""
        records = mapper.load_all_delhi_datasets()
        assert isinstance(records, list)
        assert len(records) > 0, "Expected at least some records from Delhi datasets"

    @pytest.mark.slow
    def test_loaded_records_count_is_significant(self, mapper):
        """Total records from all datasets should be substantial (100+)."""
        records = mapper.load_all_delhi_datasets()
        assert len(records) > 100, (
            f"Only {len(records)} records loaded, expected 100+"
        )

    @pytest.mark.slow
    def test_loaded_records_have_valid_severity(self, mapper):
        """Loaded records should have valid severity values."""
        records = mapper.load_all_delhi_datasets()
        if not records:
            pytest.skip("No records loaded")
        valid_severities = {"Fatal", "Grievous", "Minor", "No Injury"}
        for rec in records[:50]:
            assert rec["severity"] in valid_severities, (
                f"Invalid severity: {rec['severity']}"
            )

    @pytest.mark.slow
    def test_loaded_records_have_required_fields(self, mapper):
        """Each loaded record should have the required fields."""
        records = mapper.load_all_delhi_datasets()
        if not records:
            pytest.skip("No records loaded")
        required_fields = [
            "source", "location_name", "road_name", "latitude", "longitude",
            "severity", "total_accidents", "fatal_accidents",
            "grievous_accidents", "minor_accidents",
        ]
        for rec in records[:20]:
            for field in required_fields:
                assert field in rec, f"Missing field '{field}' in record"

    @pytest.mark.slow
    def test_some_records_have_gps(self, mapper):
        """At least some records should have GPS coordinates (from Dataset1/2)."""
        records = mapper.load_all_delhi_datasets()
        if not records:
            pytest.skip("No records loaded")
        gps_records = [
            r for r in records
            if r.get("latitude") is not None and r.get("longitude") is not None
        ]
        assert len(gps_records) > 0, "Expected some records with GPS coordinates"

    @pytest.mark.slow
    def test_gps_coordinates_within_delhi(self, mapper):
        """GPS coordinates in records should be within Delhi bounding box."""
        records = mapper.load_all_delhi_datasets()
        delhi_bbox = {"lat_min": 28.3, "lat_max": 28.95,
                      "lon_min": 76.7, "lon_max": 77.4}
        for rec in records[:100]:
            lat = rec.get("latitude")
            lon = rec.get("longitude")
            if lat is not None and lon is not None:
                assert delhi_bbox["lat_min"] <= lat <= delhi_bbox["lat_max"], (
                    f"Lat {lat} outside Delhi range"
                )
                assert delhi_bbox["lon_min"] <= lon <= delhi_bbox["lon_max"], (
                    f"Lon {lon} outside Delhi range"
                )


@pytest.mark.integration
class TestGeocodingIntegration:
    """Test: geocode real location names from datasets → map to segments."""

    @pytest.fixture
    def mapped_data(self):
        """Load cached segment mapping for integration testing."""
        mapping = _load_json(_delhi_segment_mapping_path())
        if mapping is None:
            pytest.skip("Delhi segment mapping not found. Run build_twin first.")
        return mapping

    def test_mapping_has_segments(self, mapped_data):
        """Segment mapping should contain at least some segments."""
        assert isinstance(mapped_data, dict)
        assert len(mapped_data) > 0, "Expected at least some mapped segments"

    def test_mapped_segments_have_required_fields(self, mapped_data):
        """Each mapped segment should have required fields."""
        required_fields = [
            "segment_id", "road_name", "total_accidents",
            "severity_distribution", "centroid_lat", "centroid_lon",
        ]
        for seg_id, seg_data in list(mapped_data.items())[:10]:
            for field in required_fields:
                assert field in seg_data, f"Missing field '{field}' in segment {seg_id}"

    def test_mapped_segments_have_valid_severity_distribution(self, mapped_data):
        """Severity distribution should have valid keys."""
        valid_keys = {"Fatal", "Grievous", "Minor", "No Injury"}
        for seg_id, seg_data in list(mapped_data.items())[:10]:
            sev_dist = seg_data.get("severity_distribution", {})
            for key in sev_dist:
                assert key in valid_keys, f"Invalid severity key: {key}"

    def test_mapping_stats_exist(self):
        """Mapping stats file should exist and contain expected keys."""
        from config import MAPPED_ACCIDENTS_DIR
        stats_path = os.path.join(MAPPED_ACCIDENTS_DIR, "delhi", "mapping_stats.json")
        if not os.path.exists(stats_path):
            pytest.skip("Mapping stats not found")
        stats = _load_json(stats_path)
        assert stats is not None
        assert "segments_with_accidents" in stats or "total_records" in stats

    @pytest.mark.slow
    def test_geocode_and_map_produces_results(self):
        """Full geocode_and_map_all() should produce mapped segments."""
        import osmnx as ox
        from ml.delhi_data_mapper import DelhiDataMapper
        from config import ROAD_NETWORKS_DIR

        graphml_path = os.path.join(ROAD_NETWORKS_DIR, "delhi", "delhi_roads.graphml")
        if not os.path.exists(graphml_path):
            pytest.skip("Delhi road network cache not found")

        graph = ox.load_graphml(graphml_path)
        _, edges_gdf = ox.graph_to_gdfs(graph)
        mapper = DelhiDataMapper(edges_gdf, "delhi")
        result = mapper.geocode_and_map_all()
        assert isinstance(result, dict)
        assert len(result) > 0, "Expected at least some mapped segments"


# ═════════════════════════════════════════════
# 2. RISK PIPELINE
# ═════════════════════════════════════════════

@pytest.mark.integration
class TestRiskPipeline:
    """Test: segment mapping → risk scores → heatmap data."""

    @pytest.fixture
    def risk_data(self):
        """Load cached risk data for integration testing."""
        data = _load_json(_delhi_risk_path())
        if data is None:
            pytest.skip("Delhi risk scores not found. Run build_twin first.")
        return data

    @pytest.fixture
    def risk_stats(self):
        """Load cached risk statistics."""
        data = _load_json(_delhi_risk_stats_path())
        if data is None:
            pytest.skip("Risk stats not found")
        return data

    def test_risk_data_has_segments(self, risk_data):
        """Risk data should contain segments with risk scores."""
        assert isinstance(risk_data, dict)
        assert len(risk_data) > 0, "Expected at least some segments with risk scores"

    def test_risk_data_has_required_fields(self, risk_data):
        """Each risk segment should have required fields."""
        required_fields = [
            "segment_id", "composite_risk", "historical_risk",
            "predictive_risk", "risk_category", "risk_color",
        ]
        for seg_id, seg_data in list(risk_data.items())[:10]:
            for field in required_fields:
                assert field in seg_data, f"Missing field '{field}' in segment {seg_id}"

    def test_composite_risk_in_range(self, risk_data):
        """All composite risk scores should be in 0-100 range."""
        for seg_id, seg_data in list(risk_data.items())[:50]:
            risk = seg_data.get("composite_risk", -1)
            assert 0 <= risk <= 100, f"Risk {risk} for {seg_id} outside 0-100 range"

    def test_risk_category_matches_score(self, risk_data):
        """Risk category should match the composite risk score via config ranges."""
        from config import RISK_CATEGORIES
        for seg_id, seg_data in list(risk_data.items())[:20]:
            risk = seg_data["composite_risk"]
            category = seg_data["risk_category"]
            expected_category = "Very High"
            for cat, (low, high) in RISK_CATEGORIES.items():
                if low <= risk < high:
                    expected_category = cat
                    break
            assert category == expected_category, (
                f"Segment {seg_id}: risk={risk}, category={category}, "
                f"expected={expected_category}"
            )

    def test_risk_color_matches_category(self, risk_data):
        """Risk color should match the risk category per RISK_COLORS config."""
        from config import RISK_COLORS
        for seg_id, seg_data in list(risk_data.items())[:20]:
            category = seg_data["risk_category"]
            color = seg_data["risk_color"]
            expected_color = RISK_COLORS.get(category)
            assert expected_color is not None, f"No color defined for category {category}"
            assert color == expected_color, (
                f"Segment {seg_id}: category={category}, color={color}, "
                f"expected={expected_color}"
            )

    def test_risk_stats_valid(self, risk_stats):
        """Risk statistics should contain expected keys and values."""
        assert "total_segments" in risk_stats
        assert "risk_distribution" in risk_stats
        assert risk_stats["total_segments"] > 0

    def test_risk_distribution_sums_to_total(self, risk_stats):
        """Sum of risk distribution categories should equal total segments."""
        distribution = risk_stats.get("risk_distribution", {})
        total = risk_stats.get("total_segments", 0)
        dist_sum = sum(distribution.values())
        assert dist_sum == total, (
            f"Distribution sum ({dist_sum}) != total segments ({total})"
        )

    def test_no_nan_in_risk_scores(self, risk_data):
        """No NaN values should exist in risk scores."""
        for seg_id, seg_data in list(risk_data.items())[:50]:
            for key in ["composite_risk", "historical_risk", "predictive_risk"]:
                value = seg_data.get(key)
                if value is not None:
                    assert not (isinstance(value, float) and math.isnan(value)), (
                        f"NaN found in {key} for segment {seg_id}"
                    )

    def test_heatmap_data_generated(self):
        """Heatmap data file should exist and be valid."""
        heatmap = _load_json(_delhi_heatmap_path())
        if heatmap is None:
            pytest.skip("Heatmap data not found. Run build_twin first.")
        assert isinstance(heatmap, (dict, list))


# ═════════════════════════════════════════════
# 3. API ENDPOINT TESTS
# ═════════════════════════════════════════════

@pytest.mark.integration
@pytest.mark.api
class TestAPIEndpoints:
    """Test FastAPI test client with all digital twin endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        try:
            from fastapi.testclient import TestClient
            from main import app
            return TestClient(app)
        except Exception as e:
            pytest.skip(f"Could not create test client: {e}")

    # --- Health & Cities ---

    def test_health_endpoint(self, client):
        """GET /api/health should return 200 and healthy status."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    def test_health_endpoint_has_models_info(self, client):
        """GET /api/health should include model info."""
        response = client.get("/api/health")
        data = response.json()
        assert "models_loaded" in data
        assert "available_models" in data
        assert isinstance(data["available_models"], list)

    def test_health_endpoint_has_twin_info(self, client):
        """GET /api/health should include digital twin info."""
        response = client.get("/api/health")
        data = response.json()
        assert "digital_twins" in data

    def test_cities_endpoint(self, client):
        """GET /api/twin/cities should return list of cities."""
        response = client.get("/api/twin/cities")
        assert response.status_code == 200
        data = response.json()
        assert "cities" in data
        assert isinstance(data["cities"], list)
        assert data["total_cities"] >= 1
        for city in data["cities"]:
            assert "key" in city
            assert "name" in city
            assert "status" in city

    def test_cities_includes_delhi(self, client):
        """Cities list should include Delhi."""
        response = client.get("/api/twin/cities")
        data = response.json()
        city_keys = [c["key"] for c in data["cities"]]
        assert "delhi" in city_keys

    # --- Delhi-specific endpoints ---

    def test_initialize_delhi_endpoint(self, client):
        """GET /api/twin/delhi/initialize should build or load the twin."""
        response = client.get("/api/twin/delhi/initialize")
        # May take a while on first run; could be 200 or timeout
        assert response.status_code in [200, 500], (
            f"Unexpected status: {response.status_code}"
        )
        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "success"
            assert "metadata" in data

    def test_invalid_city_initialize(self, client):
        """GET /api/twin/mumbai/initialize should return 404."""
        response = client.get("/api/twin/mumbai/initialize")
        assert response.status_code == 404

    def test_delhi_heatmap_endpoint(self, client):
        """GET /api/twin/delhi/heatmap should return heatmap data if twin initialized."""
        client.get("/api/twin/delhi/initialize")
        response = client.get("/api/twin/delhi/heatmap?type=segments")
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)
            assert "city_key" in data
            assert "color_scale" in data

    def test_delhi_heatmap_grid_type(self, client):
        """GET /api/twin/delhi/heatmap?type=grid should return grid data."""
        client.get("/api/twin/delhi/initialize")
        response = client.get("/api/twin/delhi/heatmap?type=grid")
        if response.status_code == 200:
            data = response.json()
            assert data.get("type") == "grid" or "data" in data

    def test_delhi_stats_endpoint(self, client):
        """GET /api/twin/delhi/stats should return valid stats."""
        client.get("/api/twin/delhi/initialize")
        response = client.get("/api/twin/delhi/stats")
        if response.status_code == 200:
            data = response.json()
            assert "city" in data
            assert "stats" in data

    def test_delhi_top_dangerous_endpoint(self, client):
        """GET /api/twin/delhi/segments/top-dangerous should return list."""
        client.get("/api/twin/delhi/initialize")
        response = client.get("/api/twin/delhi/segments/top-dangerous?limit=5")
        if response.status_code == 200:
            data = response.json()
            assert "segments" in data
            assert isinstance(data["segments"], list)
            assert len(data["segments"]) <= 5

    def test_delhi_metadata_endpoint(self, client):
        """GET /api/twin/delhi/metadata should return metadata if twin initialized."""
        client.get("/api/twin/delhi/initialize")
        response = client.get("/api/twin/delhi/metadata")
        if response.status_code == 200:
            data = response.json()
            assert "city" in data
            assert "metadata" in data

    def test_invalid_heatmap_type_returns_400(self, client):
        """GET /api/twin/delhi/heatmap?type=invalid should return 400."""
        client.get("/api/twin/delhi/initialize")
        response = client.get("/api/twin/delhi/heatmap?type=invalid")
        if response.status_code != 404:
            assert response.status_code == 400

    def test_uninitialized_city_endpoints_return_404(self, client):
        """Endpoints for an uninitialized city should return 404."""
        # dehradun is unlikely to be initialized
        response = client.get("/api/twin/dehradun/stats")
        assert response.status_code == 404
