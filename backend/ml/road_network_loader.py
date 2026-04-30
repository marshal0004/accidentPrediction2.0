# backend/ml/road_network_loader.py

import os
import json
import time
import logging
from datetime import datetime, timedelta

import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np

from config import ROAD_NETWORKS_DIR, CITIES_CONFIG, OSM_CACHE_EXPIRY_DAYS

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# OSMNX SETTINGS
# ─────────────────────────────────────────────
ox.settings.log_console = False
ox.settings.use_cache = True
ox.settings.timeout = 300


class RoadNetworkLoader:
    """
    Downloads, caches, and loads road networks
    from OpenStreetMap using OSMnx.
    """

    def __init__(self, city_key: str):
        """
        Initialize loader for a specific city.

        Args:
            city_key: Key from CITIES_CONFIG
                      e.g. 'delhi', 'dehradun'
        """
        if city_key not in CITIES_CONFIG:
            raise ValueError(
                f"City '{city_key}' not found in CITIES_CONFIG. "
                f"Available: {list(CITIES_CONFIG.keys())}"
            )

        self.city_key = city_key
        self.city_config = CITIES_CONFIG[city_key]
        self.city_name = self.city_config["display_name"]
        self.osm_query = self.city_config["osm_query"]
        self.network_type = self.city_config["network_type"]

        # File paths
        self.cache_dir = os.path.join(ROAD_NETWORKS_DIR, city_key)
        self.graphml_path = os.path.join(self.cache_dir, f"{city_key}_roads.graphml")
        self.metadata_path = os.path.join(self.cache_dir, f"{city_key}_metadata.json")
        self.edges_path = os.path.join(self.cache_dir, f"{city_key}_edges.csv")

        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)

        # Network object (loaded lazily)
        self._graph = None
        self._edges_gdf = None
        self._nodes_gdf = None

        logger.info(f"RoadNetworkLoader initialized for: {self.city_name}")

    # ─────────────────────────────────────────
    # CACHE MANAGEMENT
    # ─────────────────────────────────────────

    def is_cache_valid(self) -> bool:
        """
        Check if cached network exists and is not expired.

        Returns:
            True if cache is valid, False otherwise
        """
        if not os.path.exists(self.graphml_path):
            logger.info(f"No cache found for {self.city_key}")
            return False

        if not os.path.exists(self.metadata_path):
            logger.info(f"No metadata found for {self.city_key}")
            return False

        # Check expiry
        try:
            with open(self.metadata_path, "r") as f:
                metadata = json.load(f)

            cached_at = datetime.fromisoformat(metadata.get("cached_at", "2000-01-01"))
            expiry = cached_at + timedelta(days=OSM_CACHE_EXPIRY_DAYS)

            if datetime.now() > expiry:
                logger.info(
                    f"Cache expired for {self.city_key} "
                    f"(cached at {cached_at.date()})"
                )
                return False

            logger.info(
                f"Valid cache found for {self.city_key} "
                f"(cached at {cached_at.date()})"
            )
            return True

        except Exception as e:
            logger.warning(f"Cache metadata error: {e}")
            return False

    def _save_metadata(self, graph):
        """Save metadata about the downloaded network."""
        try:
            nodes, edges = ox.graph_to_gdfs(graph)
            metadata = {
                "city_key": self.city_key,
                "city_name": self.city_name,
                "osm_query": self.osm_query,
                "cached_at": datetime.now().isoformat(),
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "bbox": self.city_config["bbox"],
                "center": self.city_config["center"],
                "network_type": self.network_type,
                "crs": "EPSG:4326",
            }
            with open(self.metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Metadata saved: {self.metadata_path}")
        except Exception as e:
            logger.warning(f"Could not save metadata: {e}")

    # ─────────────────────────────────────────
    # DOWNLOAD
    # ─────────────────────────────────────────

    def download_network(self, retries: int = 3) -> nx.MultiDiGraph:
        """
        Download road network from OpenStreetMap.

        Args:
            retries: Number of retry attempts on failure

        Returns:
            NetworkX MultiDiGraph of road network
        """
        logger.info(
            f"Downloading road network for {self.city_name} " f"from OpenStreetMap..."
        )
        logger.info("This may take 1-3 minutes for large cities...")

        last_error = None

        for attempt in range(1, retries + 1):
            try:
                logger.info(f"Attempt {attempt}/{retries}...")
                start_time = time.time()

                graph = ox.graph_from_place(
                    self.osm_query, network_type=self.network_type, simplify=True
                )

                elapsed = time.time() - start_time
                nodes_count = len(graph.nodes)
                edges_count = len(graph.edges)

                logger.info(
                    f"Download complete in {elapsed:.1f}s: "
                    f"{nodes_count:,} nodes, {edges_count:,} edges"
                )
                return graph

            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt} failed: {e}")
                if attempt < retries:
                    wait = attempt * 10
                    logger.info(f"Retrying in {wait} seconds...")
                    time.sleep(wait)

        raise ConnectionError(
            f"Failed to download network for {self.city_name} "
            f"after {retries} attempts. Last error: {last_error}"
        )

    # ─────────────────────────────────────────
    # CACHE OPERATIONS
    # ─────────────────────────────────────────

    def cache_network(self, graph: nx.MultiDiGraph):
        """
        Save network to GraphML file for future use.

        Args:
            graph: NetworkX graph to cache
        """
        try:
            logger.info(f"Caching network to {self.graphml_path}...")
            ox.save_graphml(graph, self.graphml_path)
            self._save_metadata(graph)

            # Also save edges as CSV for quick loading
            self._save_edges_csv(graph)

            size_mb = os.path.getsize(self.graphml_path) / (1024 * 1024)
            logger.info(f"Network cached: {size_mb:.1f} MB")

        except Exception as e:
            logger.error(f"Failed to cache network: {e}")
            raise

    def _save_edges_csv(self, graph: nx.MultiDiGraph):
        """Save edges as CSV for fast loading."""
        try:
            nodes_gdf, edges_gdf = ox.graph_to_gdfs(graph)

            # Select useful columns only
            keep_cols = []
            possible_cols = [
                "name",
                "highway",
                "length",
                "maxspeed",
                "lanes",
                "oneway",
                "geometry",
            ]
            for col in possible_cols:
                if col in edges_gdf.columns:
                    keep_cols.append(col)

            edges_subset = edges_gdf[keep_cols].copy()

            # Convert geometry to WKT string for CSV
            edges_subset["geometry_wkt"] = edges_subset["geometry"].apply(
                lambda g: g.wkt if g is not None else ""
            )

            # Drop geometry column (can't save to CSV directly)
            edges_subset = edges_subset.drop(columns=["geometry"])

            # Reset index to get u, v, key as columns
            edges_subset = edges_subset.reset_index()

            edges_subset.to_csv(self.edges_path, index=False)
            logger.info(f"Edges CSV saved: {self.edges_path}")

        except Exception as e:
            logger.warning(f"Could not save edges CSV: {e}")

    def load_cached_network(self) -> nx.MultiDiGraph:
        """
        Load network from cached GraphML file.

        Returns:
            NetworkX MultiDiGraph
        """
        logger.info(f"Loading cached network from {self.graphml_path}...")
        start_time = time.time()

        graph = ox.load_graphml(self.graphml_path)

        elapsed = time.time() - start_time
        logger.info(
            f"Network loaded in {elapsed:.1f}s: "
            f"{len(graph.nodes):,} nodes, "
            f"{len(graph.edges):,} edges"
        )
        return graph

    # ─────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────

    def get_or_download_network(self) -> nx.MultiDiGraph:
        """
        Main entry point: load from cache or download.

        Returns:
            NetworkX MultiDiGraph of road network
        """
        if self.is_cache_valid():
            graph = self.load_cached_network()
        else:
            graph = self.download_network()
            self.cache_network(graph)

        self._graph = graph
        return graph

    # ─────────────────────────────────────────
    # GEODATAFRAMES
    # ─────────────────────────────────────────

    def get_edges_gdf(self) -> gpd.GeoDataFrame:
        """
        Get road edges as GeoDataFrame.

        Returns:
            GeoDataFrame with road segment geometries
        """
        if self._graph is None:
            self.get_or_download_network()

        if self._edges_gdf is None:
            logger.info("Converting graph to GeoDataFrame...")
            self._nodes_gdf, self._edges_gdf = ox.graph_to_gdfs(self._graph)
            logger.info(
                f"GeoDataFrame ready: " f"{len(self._edges_gdf):,} road segments"
            )

        return self._edges_gdf

    def get_nodes_gdf(self) -> gpd.GeoDataFrame:
        """
        Get road nodes as GeoDataFrame.

        Returns:
            GeoDataFrame with road node geometries
        """
        if self._graph is None:
            self.get_or_download_network()

        if self._nodes_gdf is None:
            self._nodes_gdf, self._edges_gdf = ox.graph_to_gdfs(self._graph)

        return self._nodes_gdf

    def get_metadata(self) -> dict:
        """
        Get metadata about the loaded network.

        Returns:
            Dict with network statistics
        """
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                return json.load(f)

        if self._graph is not None:
            return {
                "city_key": self.city_key,
                "city_name": self.city_name,
                "total_nodes": len(self._graph.nodes),
                "total_edges": len(self._graph.edges),
            }

        return {
            "city_key": self.city_key,
            "city_name": self.city_name,
            "status": "not_loaded",
        }

    def get_network_stats(self) -> dict:
        """
        Get detailed statistics about the road network.

        Returns:
            Dict with network statistics
        """
        edges_gdf = self.get_edges_gdf()

        stats = {
            "total_segments": len(edges_gdf),
            "total_length_km": round(edges_gdf["length"].sum() / 1000, 2),
            "avg_segment_length_m": round(edges_gdf["length"].mean(), 2),
        }

        # Road type distribution
        if "highway" in edges_gdf.columns:
            highway_counts = {}
            for val in edges_gdf["highway"]:
                if isinstance(val, list):
                    key = val[0] if val else "unknown"
                else:
                    key = str(val) if pd.notna(val) else "unknown"
                highway_counts[key] = highway_counts.get(key, 0) + 1
            stats["road_type_distribution"] = highway_counts

        return stats
