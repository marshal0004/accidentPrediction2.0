# backend/ml/heatmap_generator.py

import os
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from config import (
    DIGITAL_TWIN_DIR,
    HEATMAP_GRID_SIZE,
    RISK_CATEGORIES,
    RISK_COLORS,
    CITIES_CONFIG,
)

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HeatmapGenerator:
    """
    Generate heatmap data for risk visualization.

    Produces two types of heatmaps:
    1. Grid-based: Regular grid with risk intensity
    2. Segment-based: Road segments colored by risk
    """

    def __init__(self, edges_gdf: gpd.GeoDataFrame,
                 segment_risks: dict, city_key: str = "delhi"):
        """
        Initialize heatmap generator.

        Args:
            edges_gdf: GeoDataFrame of road network edges
            segment_risks: Dict from SegmentRiskCalculator
            city_key: City identifier
        """
        self.edges_gdf = edges_gdf.copy()
        self.segment_risks = segment_risks
        self.city_key = city_key
        self.city_config = CITIES_CONFIG[city_key]

        # Output paths
        self.output_dir = os.path.join(DIGITAL_TWIN_DIR, city_key)
        os.makedirs(self.output_dir, exist_ok=True)

        self.grid_heatmap_path = os.path.join(
            self.output_dir, "heatmap_grid.json"
        )
        self.segment_heatmap_path = os.path.join(
            self.output_dir, "heatmap_segments.json"
        )
        self.folium_map_path = os.path.join(
            self.output_dir, "heatmap_test.html"
        )

        # Get bounding box
        self.bbox = self.city_config["bbox"]
        self.center = self.city_config["center"]

        # Enrich edges with risk data
        self._enrich_edges_with_risk()

        logger.info(
            f"HeatmapGenerator initialized for {city_key}: "
            f"{len(segment_risks):,} segments with risk data"
        )

    def _enrich_edges_with_risk(self):
        """
        Add risk data to edges GeoDataFrame.
        """
        # Add segment_id if not present
        # OSMnx stores u, v, key as MultiIndex NOT as columns
        if "segment_id" not in self.edges_gdf.columns:
            if "u" in self.edges_gdf.columns and "v" in self.edges_gdf.columns:
                # u, v, key are regular columns
                key_col = self.edges_gdf["key"] if "key" in self.edges_gdf.columns else pd.Series([0]*len(self.edges_gdf), index=self.edges_gdf.index)
                self.edges_gdf["segment_id"] = (
                    self.edges_gdf["u"].astype(str) + "_" +
                    self.edges_gdf["v"].astype(str) + "_" +
                    key_col.astype(str)
                )
            elif hasattr(self.edges_gdf.index, "names") and "u" in (self.edges_gdf.index.names or []):
                # OSMnx default: (u, v, key) MultiIndex
                self.edges_gdf["segment_id"] = [
                    f"{u}_{v}_{k}"
                    for u, v, k in self.edges_gdf.index
                ]
            else:
                self.edges_gdf["segment_id"] = [
                    f"seg_{i}" for i in range(len(self.edges_gdf))
                ]

        # Add risk columns
        self.edges_gdf["risk_score"] = 0.0
        self.edges_gdf["risk_category"] = "No Risk"
        self.edges_gdf["risk_color"] = RISK_COLORS["No Risk"]

        for idx, row in self.edges_gdf.iterrows():
            segment_id = row["segment_id"]
            if segment_id in self.segment_risks:
                risk_data = self.segment_risks[segment_id]
                self.edges_gdf.at[idx, "risk_score"] = risk_data.get(
                    "composite_risk", 0
                )
                self.edges_gdf.at[idx, "risk_category"] = risk_data.get(
                    "risk_category", "No Risk"
                )
                self.edges_gdf.at[idx, "risk_color"] = risk_data.get(
                    "risk_color", RISK_COLORS["No Risk"]
                )

        logger.info("Edges enriched with risk data")

    # ─────────────────────────────────────────
    # GRID-BASED HEATMAP
    # ─────────────────────────────────────────

    def generate_grid_heatmap(self) -> list:
        """
        Generate grid-based heatmap data.

        Creates a regular grid and samples risk from
        nearby road segments.

        Returns:
            List of [lat, lon, intensity] for heatmap.js
        """
        logger.info("Generating grid-based heatmap...")

        lat_min = self.bbox["lat_min"]
        lat_max = self.bbox["lat_max"]
        lon_min = self.bbox["lon_min"]
        lon_max = self.bbox["lon_max"]

        # Create grid
        grid_size = HEATMAP_GRID_SIZE
        lats = np.linspace(lat_min, lat_max, grid_size)
        lons = np.linspace(lon_min, lon_max, grid_size)

        # OPTIMIZATION: Only check segments with risk data (1,360 vs 24,399)
        risk_segments = self.edges_gdf[
            self.edges_gdf["segment_id"].isin(self.segment_risks.keys())
        ].copy()

        logger.info(f"Using {len(risk_segments)} segments with risk data for grid")

        grid_points = []

        # Sample risk at each grid point
        total_points = grid_size * grid_size
        processed = 0

        for lat in lats:
            for lon in lons:
                processed += 1
                if processed % 500 == 0:
                    logger.info(f"Processing grid point {processed}/{total_points}...")

                # Find nearby segments (within ~5km)
                nearby_risks = []

                # CHANGED: Only iterate over segments with accidents
                for _, row in risk_segments.iterrows():
                    # Get segment centroid
                    centroid = row.get("geometry").centroid
                    seg_lat = centroid.y
                    seg_lon = centroid.x

                    # Simple distance check (approximate)
                    dist_lat = abs(seg_lat - lat)
                    dist_lon = abs(seg_lon - lon)

                    # ~0.05 degrees ≈ 5km
                    if dist_lat < 0.05 and dist_lon < 0.05:
                        nearby_risks.append(row["risk_score"])

                # Average nearby risks (or use 0 if no segments nearby)
                if nearby_risks:
                    avg_risk = np.mean(nearby_risks)
                else:
                    avg_risk = 0.0

                # Normalize to 0-1 for heatmap.js
                intensity = avg_risk / 100.0

                # Always add the point (even if 0)
                grid_points.append([lat, lon, intensity])

        logger.info(
            f"Grid heatmap generated: {len(grid_points):,} points"
        )
        return grid_points

    # ─────────────────────────────────────────
    # SEGMENT-BASED HEATMAP
    # ─────────────────────────────────────────

    def generate_segment_heatmap(self) -> list:
        """
        Generate segment-based heatmap data.

        Returns road segments with coordinates and colors.

        Returns:
            List of segment dicts for Leaflet Polylines
        """
        logger.info("Generating segment-based heatmap...")

        segments = []

        for idx, row in self.edges_gdf.iterrows():
            try:
                segment_id = row["segment_id"]
                geometry = row["geometry"]
                risk_score = row["risk_score"]
                risk_category = row["risk_category"]
                risk_color = row["risk_color"]

                # Extract coordinates from LineString
                coords = list(geometry.coords)
                coords_list = [
                    [round(lat, 6), round(lon, 6)]
                    for lon, lat in coords
                ]

                # Road name
                name = row.get("name", None)
                if isinstance(name, list):
                    name = name[0] if name else "Unknown Road"
                elif not isinstance(name, str):
                    name = "Unknown Road"

                # Line weight based on risk
                if risk_score >= 80:
                    weight = 6
                elif risk_score >= 60:
                    weight = 5
                elif risk_score >= 40:
                    weight = 4
                else:
                    weight = 3

                segment = {
                    "segment_id": segment_id,
                    "coordinates": coords_list,
                    "risk_score": round(risk_score, 2),
                    "risk_category": risk_category,
                    "color": risk_color,
                    "weight": weight,
                    "road_name": str(name),
                    "road_type": str(row.get("highway", "unknown")),
                    "length_m": round(row.get("length", 0), 2),
                }

                # Add accident count if available
                if segment_id in self.segment_risks:
                    segment["total_accidents"] = self.segment_risks[
                        segment_id
                    ].get("total_accidents", 0)
                else:
                    segment["total_accidents"] = 0

                segments.append(segment)

            except Exception as e:
                logger.debug(f"Segment {idx} skipped: {e}")
                continue

        logger.info(
            f"Segment heatmap generated: {len(segments):,} segments"
        )
        return segments

    # ─────────────────────────────────────────
    # COLOR MAPPING
    # ─────────────────────────────────────────

    @staticmethod
    def risk_to_color(risk_score: float) -> str:
        """
        Convert risk score to hex color.

        Args:
            risk_score: Risk score (0-100)

        Returns:
            Hex color string
        """
        for category, (low, high) in RISK_CATEGORIES.items():
            if low <= risk_score < high:
                return RISK_COLORS[category]

        return RISK_COLORS["Very High"]

    @staticmethod
    def get_color_scale() -> list:
        """
        Get color scale for legend.

        Returns:
            List of (category, color, range) tuples
        """
        scale = []
        for category in ["No Risk", "Low", "Moderate", "High", "Very High"]:
            color = RISK_COLORS[category]
            risk_range = RISK_CATEGORIES[category]
            scale.append({
                "category": category,
                "color": color,
                "range_min": risk_range[0],
                "range_max": risk_range[1],
            })
        return scale

    # ─────────────────────────────────────────
    # FOLIUM MAP (FOR TESTING)
    # ─────────────────────────────────────────

    def create_folium_map(self) -> object:
        """
        Create interactive Folium map for testing.

        Returns:
            Folium Map object (requires folium installed)
        """
        try:
            import folium
            from folium.plugins import HeatMap
        except ImportError:
            logger.warning(
                "Folium not available. Skipping map generation."
            )
            return None

        logger.info("Creating Folium test map...")

        # Create base map
        m = folium.Map(
            location=self.center,
            zoom_start=self.city_config["zoom_level"],
            tiles="OpenStreetMap",
        )

        # Add segment polylines
        for idx, row in self.edges_gdf.iterrows():
            if idx % 100 == 0 and idx > 0:
                logger.info(f"Added {idx} segments to map...")

            try:
                geometry = row["geometry"]
                coords = list(geometry.coords)
                coords_list = [[lat, lon] for lon, lat in coords]

                risk_score = row["risk_score"]
                risk_color = row["risk_color"]

                # Create popup
                name = row.get("name", "Unknown")
                if isinstance(name, list):
                    name = name[0] if name else "Unknown Road"

                popup_html = (
                    "<b>" + str(name) + "</b><br>"
                    "Risk: " + str(round(float(risk_score), 1)) + "<br>"
                    "Category: " + str(row['risk_category']) + "<br>"
                    "Type: " + str(row.get('highway', 'unknown'))
                )

                # Add polyline
                folium.PolyLine(
                    coords_list,
                    color=risk_color,
                    weight=4 if risk_score >= 60 else 2,
                    opacity=0.7,
                    popup=folium.Popup(popup_html, max_width=200),
                ).add_to(m)

            except Exception as e:
                logger.debug(f"Segment {idx} skipped: {e}")
                continue

        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 180px; height: 200px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p style="margin:0;"><b>Risk Level</b></p>
        <p style="margin:0;"><i class="fa fa-square" style="color:#ff0000"></i> Very High (80-100)</p>
        <p style="margin:0;"><i class="fa fa-square" style="color:#ff8000"></i> High (60-80)</p>
        <p style="margin:0;"><i class="fa fa-square" style="color:#ffff00"></i> Moderate (40-60)</p>
        <p style="margin:0;"><i class="fa fa-square" style="color:#80ff00"></i> Low (20-40)</p>
        <p style="margin:0;"><i class="fa fa-square" style="color:#00ff00"></i> Very Low (0-20)</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        logger.info("Folium map created")
        return m

    # ─────────────────────────────────────────
    # EXPORT FOR LEAFLET
    # ─────────────────────────────────────────

    def export_for_leaflet(self) -> dict:
        """
        Export all heatmap data for React Leaflet.

        Returns:
            Dict with grid and segment heatmap data
        """
        logger.info("Exporting heatmap data for Leaflet...")

        grid_data = self.generate_grid_heatmap()
        segment_data = self.generate_segment_heatmap()

        export_data = {
            "city_key": self.city_key,
            "city_name": self.city_config["name"],
            "center": self.center,
            "zoom_level": self.city_config["zoom_level"],
            "bbox": self.bbox,
            "grid_heatmap": grid_data,
            "segment_heatmap": segment_data,
            "color_scale": self.get_color_scale(),
            "generated_at": datetime.now().isoformat(),
            "grid_points_count": len(grid_data),
            "segments_count": len(segment_data),
        }

        logger.info("Export data prepared")
        return export_data

    # ─────────────────────────────────────────
    # SAVE / LOAD
    # ─────────────────────────────────────────

    def save_heatmaps(self):
        """
        Save all heatmap data to JSON files.
        """
        logger.info("Saving heatmap data...")

        # Generate data
        grid_data = self.generate_grid_heatmap()
        segment_data = self.generate_segment_heatmap()

        # Save grid heatmap
        grid_export = {
            "city_key": self.city_key,
            "type": "grid",
            "center": self.center,
            "zoom_level": self.city_config["zoom_level"],
            "grid_size": HEATMAP_GRID_SIZE,
            "data": grid_data,
            "generated_at": datetime.now().isoformat(),
        }

        with open(self.grid_heatmap_path, "w") as f:
            json.dump(grid_export, f, indent=2)

        grid_size_kb = os.path.getsize(self.grid_heatmap_path) / 1024
        logger.info(
            f"Grid heatmap saved: {self.grid_heatmap_path} "
            f"({grid_size_kb:.1f} KB)"
        )

        # Save segment heatmap
        segment_export = {
            "city_key": self.city_key,
            "type": "segments",
            "center": self.center,
            "zoom_level": self.city_config["zoom_level"],
            "color_scale": self.get_color_scale(),
            "data": segment_data,
            "generated_at": datetime.now().isoformat(),
        }

        with open(self.segment_heatmap_path, "w") as f:
            json.dump(segment_export, f, indent=2)

        segment_size_kb = os.path.getsize(
            self.segment_heatmap_path
        ) / 1024
        logger.info(
            f"Segment heatmap saved: {self.segment_heatmap_path} "
            f"({segment_size_kb:.1f} KB)"
        )

        # Save Folium test map (optional)
        try:
            m = self.create_folium_map()
            if m is not None:
                m.save(self.folium_map_path)
                logger.info(
                    f"Folium test map saved: {self.folium_map_path}"
                )
                logger.info(
                    f"Open {self.folium_map_path} in browser to view"
                )
        except Exception as e:
            logger.warning(f"Folium map generation failed: {e}")

    def load_grid_heatmap(self) -> dict:
        """Load grid heatmap from JSON."""
        if not os.path.exists(self.grid_heatmap_path):
            logger.warning(
                f"No grid heatmap found at {self.grid_heatmap_path}"
            )
            return {}

        with open(self.grid_heatmap_path, "r") as f:
            data = json.load(f)

        logger.info(
            f"Grid heatmap loaded: {len(data.get('data', []))} points"
        )
        return data

    def load_segment_heatmap(self) -> dict:
        """Load segment heatmap from JSON."""
        if not os.path.exists(self.segment_heatmap_path):
            logger.warning(
                f"No segment heatmap found at "
                f"{self.segment_heatmap_path}"
            )
            return {}

        with open(self.segment_heatmap_path, "r") as f:
            data = json.load(f)

        logger.info(
            f"Segment heatmap loaded: "
            f"{len(data.get('data', []))} segments"
        )
        return data

    def is_heatmaps_valid(self) -> bool:
        """Check if saved heatmaps exist."""
        return (
            os.path.exists(self.grid_heatmap_path) and
            os.path.exists(self.segment_heatmap_path)
        )
