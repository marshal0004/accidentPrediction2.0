# backend/ml/accident_segment_mapper.py

import os
import json
import logging
import math
from datetime import datetime

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

from config import (
    MAPPED_ACCIDENTS_DIR,
    NH_CORRIDORS,
    SEVERITY_CODE_MAP,
    WEATHER_CODE_MAP,
    ROAD_CONDITION_CODE_MAP,
    ROAD_FEATURE_CODE_MAP,
    CAUSES_CODE_MAP,
    VEHICLE_CODE_MAP,
    ROADSIDE_CODE_MAP,
    MAX_SNAP_DISTANCE_METERS,
    PRIMARY_PATH,
    SECONDARY_PATH,
)

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# GPS CONVERSION UTILITIES
# ─────────────────────────────────────────────


def chainage_to_gps(nh_code: int, chainage_km: float) -> tuple:
    """
    Convert NH corridor code + chainage km to GPS coordinates.

    Uses linear interpolation along the NH corridor
    from start to end coordinates.

    Args:
        nh_code: NH corridor code (1 or 2)
        chainage_km: Distance from start of NH in km

    Returns:
        (latitude, longitude) tuple
    """
    if nh_code not in NH_CORRIDORS:
        logger.warning(f"Unknown NH code: {nh_code}")
        return None, None

    corridor = NH_CORRIDORS[nh_code]
    total_length = corridor["total_length_km"]

    # Clamp chainage to valid range
    chainage_km = max(0, min(chainage_km, total_length))

    # Fraction along the corridor
    fraction = chainage_km / total_length

    # Linear interpolation
    lat = corridor["start_lat"] + fraction * (
        corridor["end_lat"] - corridor["start_lat"]
    )
    lon = corridor["start_lon"] + fraction * (
        corridor["end_lon"] - corridor["start_lon"]
    )

    return round(lat, 6), round(lon, 6)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two GPS points in meters.

    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates

    Returns:
        Distance in meters
    """
    R = 6371000  # Earth radius in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def area_to_gps(area_name: str, city_center: list) -> tuple:
    """
    Convert area name string to approximate GPS coordinates.

    Maps common area descriptions to GPS offsets
    from city center.

    Args:
        area_name: Area description string
        city_center: [lat, lon] of city center

    Returns:
        (latitude, longitude) tuple
    """
    # Area name to GPS offset mapping
    # Offsets are approximate (degrees)
    area_offsets = {
        "residential": (0.05, 0.05),
        "office": (0.02, 0.02),
        "recreational": (-0.03, 0.07),
        "industrial": (-0.05, -0.05),
        "hospital": (0.04, -0.03),
        "school": (-0.02, 0.04),
        "market": (0.01, 0.01),
        "rural": (-0.08, -0.08),
        "unknown": (0.00, 0.00),
        "church": (0.03, 0.03),
        "outside": (-0.10, -0.10),
    }

    if not isinstance(area_name, str):
        return city_center[0], city_center[1]

    area_lower = area_name.lower().strip()

    # Match area name to offset
    for key, offset in area_offsets.items():
        if key in area_lower:
            lat = city_center[0] + offset[0]
            lon = city_center[1] + offset[1]
            return round(lat, 6), round(lon, 6)

    # Default: city center with small random offset
    lat = city_center[0] + np.random.uniform(-0.05, 0.05)
    lon = city_center[1] + np.random.uniform(-0.05, 0.05)
    return round(lat, 6), round(lon, 6)


# ─────────────────────────────────────────────
# MAIN MAPPER CLASS
# ─────────────────────────────────────────────


class AccidentSegmentMapper:
    """
    Maps accident records from CSV datasets to
    road network segments using GPS coordinates.

    Handles both:
    - ETP_4 dataset (NH corridor + chainage → GPS)
    - Road.csv dataset (area name → GPS)
    """

    def __init__(self, edges_gdf: gpd.GeoDataFrame, city_key: str = "delhi"):
        """
        Initialize mapper with road network edges.

        Args:
            edges_gdf: GeoDataFrame of road segments
                       from RoadNetworkLoader
            city_key: City key for GPS conversion
        """
        self.edges_gdf = edges_gdf.copy()
        self.city_key = city_key

        # Import city config here to avoid circular import
        from config import CITIES_CONFIG

        self.city_config = CITIES_CONFIG[city_key]
        self.city_center = self.city_config["center"]

        # Output paths
        self.output_dir = os.path.join(MAPPED_ACCIDENTS_DIR, city_key)
        os.makedirs(self.output_dir, exist_ok=True)

        self.mapping_path = os.path.join(self.output_dir, "segment_mapping.json")
        self.stats_path = os.path.join(self.output_dir, "mapping_stats.json")

        # Results storage
        self.segment_mapping = {}
        self.mapping_stats = {
            "total_accidents": 0,
            "mapped_accidents": 0,
            "unmapped_accidents": 0,
            "mapping_rate_pct": 0.0,
            "segments_with_accidents": 0,
            "mapped_at": datetime.now().isoformat(),
        }

        # Prepare edges for spatial lookup
        self._prepare_edges()

        logger.info(
            f"AccidentSegmentMapper initialized: "
            f"{len(self.edges_gdf):,} road segments"
        )

    def _prepare_edges(self):
        """
        Prepare edge GeoDataFrame for fast spatial lookup.
        Adds centroid coordinates to each edge.
        """
        try:
            # Reproject to metric CRS for distance calculations
            edges_metric = self.edges_gdf.to_crs("EPSG:3857")
            self.edges_metric = edges_metric

            # Calculate centroids in WGS84
            centroids = self.edges_gdf.geometry.centroid
            self.edge_centroids_lat = centroids.y.values
            self.edge_centroids_lon = centroids.x.values

            # Create segment IDs
            self.edges_gdf = self.edges_gdf.reset_index()
            if "u" in self.edges_gdf.columns:
                self.edges_gdf["segment_id"] = (
                    self.edges_gdf["u"].astype(str)
                    + "_"
                    + self.edges_gdf["v"].astype(str)
                    + "_"
                    + self.edges_gdf.get(
                        "key", pd.Series([0] * len(self.edges_gdf))
                    ).astype(str)
                )
            else:
                self.edges_gdf["segment_id"] = [
                    f"seg_{i}" for i in range(len(self.edges_gdf))
                ]

            logger.info("Edge preparation complete")

        except Exception as e:
            logger.error(f"Edge preparation failed: {e}")
            raise

    def _find_nearest_segment(self, lat: float, lon: float) -> tuple:
        """
        Find nearest road segment to a GPS point.

        Uses vectorized haversine distance calculation
        for performance.

        Args:
            lat: Latitude of accident
            lon: Longitude of accident

        Returns:
            (segment_id, distance_meters) tuple
            Returns (None, None) if no segment within
            MAX_SNAP_DISTANCE_METERS
        """
        if lat is None or lon is None:
            return None, None

        if np.isnan(lat) or np.isnan(lon):
            return None, None

        # Vectorized distance calculation
        R = 6371000
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)

        cent_lat_rad = np.radians(self.edge_centroids_lat)
        cent_lon_rad = np.radians(self.edge_centroids_lon)

        dphi = cent_lat_rad - lat_rad
        dlambda = cent_lon_rad - lon_rad

        a = (
            np.sin(dphi / 2) ** 2
            + math.cos(lat_rad) * np.cos(cent_lat_rad) * np.sin(dlambda / 2) ** 2
        )

        distances = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        # Find nearest
        nearest_idx = int(np.argmin(distances))
        min_distance = float(distances[nearest_idx])

        if min_distance > MAX_SNAP_DISTANCE_METERS:
            return None, None

        segment_id = self.edges_gdf.iloc[nearest_idx]["segment_id"]
        return segment_id, min_distance

    def _get_segment_info(self, idx: int) -> dict:
        """
        Get road segment information at a given index.

        Args:
            idx: Index in edges_gdf

        Returns:
            Dict with segment details
        """
        row = self.edges_gdf.iloc[idx]

        # Road name
        name = row.get("name", None)
        if isinstance(name, list):
            name = name[0] if name else "Unknown Road"
        elif not isinstance(name, str):
            name = "Unknown Road"

        # Road type
        highway = row.get("highway", "unknown")
        if isinstance(highway, list):
            highway = highway[0] if highway else "unknown"

        # Length
        length = float(row.get("length", 0))

        # Geometry centroid
        centroid = row.get("geometry").centroid
        lat = centroid.y
        lon = centroid.x

        return {
            "segment_id": row["segment_id"],
            "road_name": str(name),
            "road_type": str(highway),
            "length_m": round(length, 2),
            "centroid_lat": round(lat, 6),
            "centroid_lon": round(lon, 6),
        }

    # ─────────────────────────────────────────
    # ETP_4 DATASET MAPPING
    # ─────────────────────────────────────────

    def map_etp4_accidents(self) -> dict:
        """
        Map ETP_4_New_Data_Accidents.csv accidents
        to road segments using chainage-to-GPS conversion.

        Returns:
            Dict mapping segment_id to accident list
        """
        if not os.path.exists(PRIMARY_PATH):
            logger.warning(f"ETP_4 dataset not found: {PRIMARY_PATH}")
            return {}

        logger.info("Loading ETP_4 dataset...")
        df = pd.read_csv(PRIMARY_PATH)
        logger.info(f"ETP_4 records: {len(df):,}")

        mapped = {}
        mapped_count = 0
        unmapped_count = 0

        for idx, row in df.iterrows():
            try:
                # Convert chainage to GPS
                nh_code = int(row.get("Accident_Location_A", 0))
                chainage = float(row.get("Accident_Location_A_Chainage_km", 0))

                lat, lon = chainage_to_gps(nh_code, chainage)

                if lat is None or lon is None:
                    unmapped_count += 1
                    continue

                # Find nearest segment
                segment_id, distance = self._find_nearest_segment(lat, lon)

                if segment_id is None:
                    unmapped_count += 1
                    continue

                # Build accident record
                severity_code = int(row.get("Accident_Severity_C", 3))
                weather_code = int(row.get("Weather_Conditions_H", 1))
                road_cond_code = int(row.get("Road_Condition_F", 1))
                road_feat_code = int(row.get("Road_Feature_E", 1))
                cause_code = int(row.get("Causes_D", 8))

                accident = {
                    "accident_id": f"etp4_{idx}",
                    "source": "ETP4",
                    "date": str(row.get("Date", "")),
                    "day_of_week": int(row.get("Day_of_Week", 0)),
                    "time": str(row.get("Time_of_Accident", "")),
                    "severity": SEVERITY_CODE_MAP.get(severity_code, "Minor"),
                    "severity_code": severity_code,
                    "weather": WEATHER_CODE_MAP.get(weather_code, "Clear"),
                    "road_condition": ROAD_CONDITION_CODE_MAP.get(
                        road_cond_code, "Dry"
                    ),
                    "road_feature": ROAD_FEATURE_CODE_MAP.get(
                        road_feat_code, "Straight Road"
                    ),
                    "cause": CAUSES_CODE_MAP.get(cause_code, "Other"),
                    "nh_corridor": nh_code,
                    "chainage_km": chainage,
                    "roadside": ROADSIDE_CODE_MAP.get(
                        int(row.get("Accident_Location_A_Chainage_km_RoadSide", 1)),
                        "Left",
                    ),
                    "lat": lat,
                    "lon": lon,
                    "snap_distance_m": round(distance, 2),
                }

                # Add to mapping
                if segment_id not in mapped:
                    mapped[segment_id] = []
                mapped[segment_id].append(accident)
                mapped_count += 1

            except Exception as e:
                logger.debug(f"Row {idx} mapping error: {e}")
                unmapped_count += 1
                continue

        logger.info(
            f"ETP_4 mapping: {mapped_count:,} mapped, " f"{unmapped_count:,} unmapped"
        )
        return mapped

    # ─────────────────────────────────────────
    # ROAD.CSV DATASET MAPPING
    # ─────────────────────────────────────────

    def map_road_csv_accidents(self) -> dict:
        """
        Map Road.csv accidents to road segments
        using area-name-to-GPS conversion.

        Returns:
            Dict mapping segment_id to accident list
        """
        if not os.path.exists(SECONDARY_PATH):
            logger.warning(f"Road.csv not found: {SECONDARY_PATH}")
            return {}

        logger.info("Loading Road.csv dataset...")
        df = pd.read_csv(SECONDARY_PATH)
        logger.info(f"Road.csv records: {len(df):,}")

        mapped = {}
        mapped_count = 0
        unmapped_count = 0

        for idx, row in df.iterrows():
            try:
                # Convert area name to GPS
                area = str(row.get("Area_accident_occured", "unknown"))
                lat, lon = area_to_gps(area, self.city_center)

                # Find nearest segment
                segment_id, distance = self._find_nearest_segment(lat, lon)

                if segment_id is None:
                    unmapped_count += 1
                    continue

                # Parse severity
                severity_raw = str(
                    row.get("Accident_severity", "Slight Injury")
                ).lower()

                if "fatal" in severity_raw:
                    severity = "Fatal"
                elif "serious" in severity_raw or "grievous" in severity_raw:
                    severity = "Grievous"
                elif "slight" in severity_raw or "minor" in severity_raw:
                    severity = "Minor"
                else:
                    severity = "No Injury"

                accident = {
                    "accident_id": f"road_{idx}",
                    "source": "Road_CSV",
                    "time": str(row.get("Time", "")),
                    "day_of_week": str(row.get("Day_of_week", "")),
                    "severity": severity,
                    "weather": str(row.get("Weather_conditions", "Normal")),
                    "road_condition": str(row.get("Road_surface_conditions", "Dry")),
                    "road_feature": str(row.get("Road_allignment", "")),
                    "cause": str(row.get("Cause_of_accident", "")),
                    "area": area,
                    "light_condition": str(row.get("Light_conditions", "")),
                    "collision_type": str(row.get("Type_of_collision", "")),
                    "vehicles_involved": int(row.get("Number_of_vehicles_involved", 1)),
                    "lat": lat,
                    "lon": lon,
                    "snap_distance_m": round(distance, 2),
                }

                if segment_id not in mapped:
                    mapped[segment_id] = []
                mapped[segment_id].append(accident)
                mapped_count += 1

            except Exception as e:
                logger.debug(f"Row {idx} mapping error: {e}")
                unmapped_count += 1
                continue

        logger.info(
            f"Road.csv mapping: {mapped_count:,} mapped, "
            f"{unmapped_count:,} unmapped"
        )
        return mapped

    # ─────────────────────────────────────────
    # AGGREGATE AND SAVE
    # ─────────────────────────────────────────

    def aggregate_segment_data(self, mapping: dict) -> dict:
        """
        Aggregate accident data per segment into
        structured format with statistics.

        Args:
            mapping: Raw mapping dict
                     {segment_id: [accident, ...]}

        Returns:
            Enriched mapping with statistics per segment
        """
        aggregated = {}

        for segment_id, accidents in mapping.items():
            if not accidents:
                continue

            # Find segment info
            seg_rows = self.edges_gdf[self.edges_gdf["segment_id"] == segment_id]

            if len(seg_rows) == 0:
                continue

            seg_info = self._get_segment_info(seg_rows.index[0])

            # Count by severity
            severity_counts = {"Fatal": 0, "Grievous": 0, "Minor": 0, "No Injury": 0}
            for acc in accidents:
                sev = acc.get("severity", "Minor")
                if sev in severity_counts:
                    severity_counts[sev] += 1

            # Count by weather
            weather_counts = {}
            for acc in accidents:
                w = acc.get("weather", "Unknown")
                weather_counts[w] = weather_counts.get(w, 0) + 1

            # Count by time period
            time_counts = {"Morning": 0, "Afternoon": 0, "Evening": 0, "Night": 0}
            for acc in accidents:
                time_str = str(acc.get("time", ""))
                try:
                    hour = int(time_str.split(":")[0])
                    if 6 <= hour < 12:
                        time_counts["Morning"] += 1
                    elif 12 <= hour < 17:
                        time_counts["Afternoon"] += 1
                    elif 17 <= hour < 21:
                        time_counts["Evening"] += 1
                    else:
                        time_counts["Night"] += 1
                except Exception:
                    time_counts["Night"] += 1

            # Fatal rate
            total = len(accidents)
            fatal_rate = severity_counts["Fatal"] / total if total > 0 else 0

            aggregated[segment_id] = {
                "segment_id": segment_id,
                "road_name": seg_info["road_name"],
                "road_type": seg_info["road_type"],
                "length_m": seg_info["length_m"],
                "centroid_lat": seg_info["centroid_lat"],
                "centroid_lon": seg_info["centroid_lon"],
                "total_accidents": total,
                "severity_distribution": severity_counts,
                "weather_distribution": weather_counts,
                "time_distribution": time_counts,
                "fatal_rate": round(fatal_rate, 4),
                "accidents": accidents,
            }

        return aggregated

    def map_all_accidents(self) -> dict:
        """
        Map accidents from ALL datasets to segments.
        Main entry point for accident mapping.

        Returns:
            Complete aggregated segment mapping dict
        """
        logger.info("=" * 50)
        logger.info("Starting accident-to-segment mapping...")
        logger.info("=" * 50)

        combined_mapping = {}

        # Map ETP_4 accidents
        logger.info("\n[1/2] Mapping ETP_4 accidents...")
        etp4_mapping = self.map_etp4_accidents()

        for seg_id, accidents in etp4_mapping.items():
            if seg_id not in combined_mapping:
                combined_mapping[seg_id] = []
            combined_mapping[seg_id].extend(accidents)

        # Map Road.csv accidents
        logger.info("\n[2/2] Mapping Road.csv accidents...")
        road_mapping = self.map_road_csv_accidents()

        for seg_id, accidents in road_mapping.items():
            if seg_id not in combined_mapping:
                combined_mapping[seg_id] = []
            combined_mapping[seg_id].extend(accidents)

        # Total counts
        total_accidents = sum(len(v) for v in combined_mapping.values())
        logger.info(
            f"\nCombined mapping: "
            f"{total_accidents:,} accidents across "
            f"{len(combined_mapping):,} segments"
        )

        # Aggregate data per segment
        logger.info("Aggregating segment data...")
        aggregated = self.aggregate_segment_data(combined_mapping)

        # Update stats
        self.mapping_stats.update(
            {
                "total_accidents": total_accidents,
                "mapped_accidents": total_accidents,
                "segments_with_accidents": len(aggregated),
                "mapping_rate_pct": round(
                    len(aggregated) / max(len(self.edges_gdf), 1) * 100, 2
                ),
                "mapped_at": datetime.now().isoformat(),
                "etp4_segments": len(etp4_mapping),
                "road_csv_segments": len(road_mapping),
            }
        )

        self.segment_mapping = aggregated
        return aggregated

    # ─────────────────────────────────────────
    # SAVE / LOAD
    # ─────────────────────────────────────────

    def save_mapping(self, mapping: dict = None):
        """
        Save segment mapping to JSON file.

        Args:
            mapping: Mapping dict to save
                     (uses self.segment_mapping if None)
        """
        if mapping is None:
            mapping = self.segment_mapping

        # Save mapping (without full accident list
        # to keep file size manageable)
        save_data = {}
        for seg_id, data in mapping.items():
            save_data[seg_id] = {k: v for k, v in data.items() if k != "accidents"}
            # Save only last 50 accidents per segment
            save_data[seg_id]["accidents"] = data.get("accidents", [])[:50]

        with open(self.mapping_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)

        # Save stats
        with open(self.stats_path, "w") as f:
            json.dump(self.mapping_stats, f, indent=2)

        size_kb = os.path.getsize(self.mapping_path) / 1024
        logger.info(f"Mapping saved: {self.mapping_path} " f"({size_kb:.1f} KB)")
        logger.info(f"Stats saved: {self.stats_path}")

    def load_mapping(self) -> dict:
        """
        Load segment mapping from JSON file.

        Returns:
            Mapping dict or empty dict if not found
        """
        if not os.path.exists(self.mapping_path):
            logger.warning(f"No mapping found at {self.mapping_path}")
            return {}

        with open(self.mapping_path, "r") as f:
            mapping = json.load(f)

        logger.info(f"Mapping loaded: {len(mapping):,} segments")
        return mapping

    def is_mapping_valid(self) -> bool:
        """Check if saved mapping exists."""
        return os.path.exists(self.mapping_path)

    def get_stats(self) -> dict:
        """Get mapping statistics."""
        if os.path.exists(self.stats_path):
            with open(self.stats_path, "r") as f:
                return json.load(f)
        return self.mapping_stats
