# backend/ml/scenario_simulator.py

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from config import (
    DIGITAL_TWIN_DIR,
    INTERVENTIONS,
    VALUE_OF_STATISTICAL_LIFE_INR,
    BASELINE_FATALITIES_PER_HIGH_RISK_SEGMENT,
)

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScenarioSimulator:
    """
    Simulate "what-if" scenarios for road segments.

    Capabilities:
    1. Weather change simulation
    2. Time of day change simulation
    3. Traffic level change simulation
    4. Safety intervention simulation
    5. Multi-factor scenario comparison
    """

    def __init__(self, predictor, segment_risks: dict,
                 city_key: str = "delhi"):
        """
        Initialize scenario simulator.

        Args:
            predictor: AccidentPredictor instance for ML predictions
            segment_risks: Dict from SegmentRiskCalculator
            city_key: City identifier
        """
        self.predictor = predictor
        self.segment_risks = segment_risks
        self.city_key = city_key

        # Output paths
        self.output_dir = os.path.join(DIGITAL_TWIN_DIR, city_key)
        os.makedirs(self.output_dir, exist_ok=True)

        self.scenarios_path = os.path.join(
            self.output_dir, "simulated_scenarios.json"
        )

        # Simulation results storage
        self.simulated_scenarios = {}

        logger.info(
            f"ScenarioSimulator initialized for {city_key}: "
            f"{len(segment_risks):,} segments"
        )

    # ─────────────────────────────────────────
    # BASE INPUT GENERATION
    # ─────────────────────────────────────────

    def _create_base_input(self, segment_id: str) -> dict:
        """
        Create base prediction input for a segment.

        Uses typical/average conditions.

        Args:
            segment_id: Segment identifier

        Returns:
            Dict matching predictor input schema
        """
        segment_data = self.segment_risks.get(segment_id, {})

        # Default base input (typical conditions)
        base_input = {
            "Day_of_Week": "Monday",
            "Time_of_Accident": "12:00",
            "Accident_Location": segment_data.get("road_name", "Unknown"),
            "Nature_of_Accident": "Collision",
            "Causes_D1": "Other",
            "Road_Condition": "Dry",
            "Weather_Conditions": "Clear",
            "Intersection_Type": "No Junction",
            "Vehicle_Type_V1": "Car/Jeep/Van",
            "Vehicle_Type_V2": "",
            "Number_of_Vehicles": 2,
        }

        return base_input

    # ─────────────────────────────────────────
    # SCENARIO MODIFICATIONS
    # ─────────────────────────────────────────

    def _modify_weather(self, base_input: dict,
                         weather: str) -> dict:
        """
        Modify weather conditions in input.

        Args:
            base_input: Base prediction input
            weather: Weather condition
                     ("Clear", "Rain", "Fog", "Snow", etc.)

        Returns:
            Modified input dict
        """
        modified = base_input.copy()
        modified["Weather_Conditions"] = weather

        # Also modify road condition based on weather
        if weather == "Rain":
            modified["Road_Condition"] = "Wet"
        elif weather == "Snow":
            modified["Road_Condition"] = "Snow/Ice"
        elif weather == "Dust Storm":
            modified["Road_Condition"] = "Dusty"
        else:
            modified["Road_Condition"] = "Dry"

        return modified

    def _modify_time(self, base_input: dict, time_period: str) -> dict:
        """
        Modify time of day in input.

        Args:
            base_input: Base prediction input
            time_period: Time period
                         ("Morning", "Day", "Evening", "Night")

        Returns:
            Modified input dict
        """
        modified = base_input.copy()

        time_map = {
            "Morning": "08:00",
            "Day": "14:00",
            "Afternoon": "14:00",
            "Evening": "18:00",
            "Night": "23:00",
        }

        modified["Time_of_Accident"] = time_map.get(
            time_period, "12:00"
        )

        return modified

    def _modify_traffic(self, base_input: dict,
                         traffic_level: str) -> dict:
        """
        Modify traffic level in input.

        Args:
            base_input: Base prediction input
            traffic_level: Traffic level
                           ("Low", "Medium", "High")

        Returns:
            Modified input dict
        """
        modified = base_input.copy()

        # Traffic level affects number of vehicles
        traffic_map = {
            "Low": 1,
            "Medium": 2,
            "High": 3,
        }

        modified["Number_of_Vehicles"] = traffic_map.get(
            traffic_level, 2
        )

        return modified

    def _apply_intervention(self, base_input: dict,
                             intervention_id: str,
                             base_risk: float) -> tuple:
        """
        Apply safety intervention effect.

        Interventions modify risk by a percentage
        based on their expected effectiveness.

        Args:
            base_input: Base prediction input
            intervention_id: Intervention identifier
            base_risk: Base risk score before intervention

        Returns:
            (modified_input, adjusted_risk) tuple
        """
        if intervention_id not in INTERVENTIONS:
            logger.warning(
                f"Unknown intervention: {intervention_id}"
            )
            return base_input, base_risk

        intervention = INTERVENTIONS[intervention_id]

        # Get expected risk reduction range
        min_reduction = intervention["expected_risk_reduction_min"]
        max_reduction = intervention["expected_risk_reduction_max"]

        # Use average reduction
        avg_reduction_pct = (min_reduction + max_reduction) / 2

        # Calculate new risk
        reduction_factor = (100 - avg_reduction_pct) / 100
        new_risk = base_risk * reduction_factor

        # For interventions, we don't modify input features
        # (since ML model wasn't trained with intervention features)
        # Instead, we apply the reduction directly to risk score
        modified = base_input.copy()

        return modified, new_risk

    # ─────────────────────────────────────────
    # SIMULATION EXECUTION
    # ─────────────────────────────────────────

    def _predict_risk(self, input_dict: dict) -> float:
        """
        Get risk score from ML prediction or rule-based fallback.
        """
        if self.predictor is None:
            return self._rule_based_risk(input_dict)

        try:
            result = self.predictor.predict(input_dict)
            probabilities = result.get("probabilities", {})

            weighted_risk = (
                probabilities.get("Fatal", 0) * 100 +
                probabilities.get("Grievous", 0) * 75 +
                probabilities.get("Minor", 0) * 50 +
                probabilities.get("No Injury", 0) * 25
            )
            return round(weighted_risk, 2)

        except Exception as e:
            logger.warning(f"ML prediction failed, using rules: {e}")
            return self._rule_based_risk(input_dict)

    def _rule_based_risk(self, input_dict: dict) -> float:
        """
        Rule-based risk scoring when ML predictor unavailable.
        Based on known road safety risk factors.
        """
        risk = 40.0

        # Weather effect
        weather = input_dict.get("Weather_Conditions", "Clear")
        weather_risk = {
            "Clear": 0, "Cloudy": 3, "Rain": 15,
            "Heavy Rain": 22, "Fog": 20, "Dense Fog": 28,
            "Snow": 18, "Dust Storm": 12, "Hail": 16,
        }
        risk += weather_risk.get(weather, 0)

        # Road condition effect
        road = input_dict.get("Road_Condition", "Dry")
        road_risk = {
            "Dry": 0, "Wet": 12, "Snow/Ice": 20,
            "Potholed": 15, "Muddy": 10,
            "Dusty": 8, "Under Construction": 18,
        }
        risk += road_risk.get(road, 0)

        # Time of day effect
        time_str = input_dict.get("Time_of_Accident", "12:00")
        try:
            hour = int(time_str.split(":")[0])
        except Exception:
            hour = 12

        if 22 <= hour or hour < 6:
            risk += 18    # Late night
        elif 6 <= hour < 9:
            risk += 8     # Morning rush
        elif 17 <= hour < 20:
            risk += 10    # Evening rush
        else:
            risk += 0     # Daytime base

        # Traffic effect
        vehicles = input_dict.get("Number_of_Vehicles", 2)
        if vehicles >= 3:
            risk += 10
        elif vehicles == 1:
            risk -= 5

        # Weekend effect
        day = input_dict.get("Day_of_Week", "Monday")
        if day in {"Saturday", "Sunday"}:
            risk += 5

        return round(max(0.0, min(100.0, risk)), 2)
  
  
  
  
    def simulate_weather_change(self, segment_id: str,
                                 weather: str) -> dict:
        """Simulate weather change on a segment."""
        segment_data = self.segment_risks.get(segment_id, {})
        historical_risk = segment_data.get("composite_risk", 0.0)

        base_input = self._create_base_input(segment_id)
        base_ml_risk = self._rule_based_risk(base_input)

        modified_input = self._modify_weather(base_input, weather)
        new_ml_risk = self._rule_based_risk(modified_input)

        ml_risk_change = new_ml_risk - base_ml_risk

        # Use historical risk as base, apply ML-derived change
        base_risk = historical_risk if historical_risk > 0 else base_ml_risk
        new_risk = max(0.0, min(100.0, base_risk + ml_risk_change))

        risk_change = new_risk - base_risk
        risk_change_pct = (
            (risk_change / base_risk * 100) if base_risk > 0 else 0
        )

        return {
            "scenario_type": "weather_change",
            "segment_id": segment_id,
            "base_weather": base_input["Weather_Conditions"],
            "new_weather": weather,
            "base_risk": round(base_risk, 2),
            "new_risk": round(new_risk, 2),
            "risk_change": round(risk_change, 2),
            "risk_change_pct": round(risk_change_pct, 2),
            "note": "Risk change based on weather impact model",
        }

    def simulate_time_change(self, segment_id: str,
                              time_period: str) -> dict:
        """Simulate time of day change on a segment."""
        segment_data = self.segment_risks.get(segment_id, {})
        historical_risk = segment_data.get("composite_risk", 0.0)

        base_input = self._create_base_input(segment_id)
        base_ml_risk = self._rule_based_risk(base_input)

        modified_input = self._modify_time(base_input, time_period)
        new_ml_risk = self._rule_based_risk(modified_input)

        ml_risk_change = new_ml_risk - base_ml_risk

        base_risk = historical_risk if historical_risk > 0 else base_ml_risk
        new_risk = max(0.0, min(100.0, base_risk + ml_risk_change))

        risk_change = new_risk - base_risk
        risk_change_pct = (
            (risk_change / base_risk * 100) if base_risk > 0 else 0
        )

        return {
            "scenario_type": "time_change",
            "segment_id": segment_id,
            "base_time": base_input["Time_of_Accident"],
            "new_time_period": time_period,
            "base_risk": round(base_risk, 2),
            "new_risk": round(new_risk, 2),
            "risk_change": round(risk_change, 2),
            "risk_change_pct": round(risk_change_pct, 2),
            "note": "Risk change based on time-of-day impact model",
        }

    def simulate_traffic_change(self, segment_id: str,
                                 traffic_level: str) -> dict:
        """Simulate traffic level change on a segment."""
        segment_data = self.segment_risks.get(segment_id, {})
        historical_risk = segment_data.get("composite_risk", 0.0)

        base_input = self._create_base_input(segment_id)
        base_ml_risk = self._rule_based_risk(base_input)

        modified_input = self._modify_traffic(base_input, traffic_level)
        new_ml_risk = self._rule_based_risk(modified_input)

        ml_risk_change = new_ml_risk - base_ml_risk

        base_risk = historical_risk if historical_risk > 0 else base_ml_risk
        new_risk = max(0.0, min(100.0, base_risk + ml_risk_change))

        risk_change = new_risk - base_risk
        risk_change_pct = (
            (risk_change / base_risk * 100) if base_risk > 0 else 0
        )

        return {
            "scenario_type": "traffic_change",
            "segment_id": segment_id,
            "base_traffic": "Medium",
            "new_traffic": traffic_level,
            "base_risk": round(base_risk, 2),
            "new_risk": round(new_risk, 2),
            "risk_change": round(risk_change, 2),
            "risk_change_pct": round(risk_change_pct, 2),
            "note": "Risk change based on traffic impact model",
        }



    def simulate_intervention(self, segment_id: str,
                               intervention_id: str) -> dict:
        """
        Simulate safety intervention on a segment.

        Args:
            segment_id: Segment identifier
            intervention_id: Intervention to simulate

        Returns:
            Simulation result dict with ROI analysis
        """
        if intervention_id not in INTERVENTIONS:
            raise ValueError(
                f"Unknown intervention: {intervention_id}. "
                f"Available: {list(INTERVENTIONS.keys())}"
            )

        intervention = INTERVENTIONS[intervention_id]

        # Get base risk from segment data
        segment_data = self.segment_risks.get(segment_id, {})
        base_risk = segment_data.get("composite_risk", 50.0)

        # Apply intervention
        base_input = self._create_base_input(segment_id)
        _, new_risk = self._apply_intervention(
            base_input, intervention_id, base_risk
        )

        # Calculate risk reduction
        risk_reduction = base_risk - new_risk
        risk_reduction_pct = (
            (risk_reduction / base_risk * 100) if base_risk > 0 else 0
        )

        # Estimate lives saved per year
        # Assumes high-risk segments have baseline fatalities per year
        if base_risk >= 60:  # High risk
            baseline_fatalities = BASELINE_FATALITIES_PER_HIGH_RISK_SEGMENT
        else:
            baseline_fatalities = BASELINE_FATALITIES_PER_HIGH_RISK_SEGMENT * (base_risk / 60)

        lives_saved_per_year = baseline_fatalities * (risk_reduction_pct / 100)

        # Calculate cost
        segment_length_km = segment_data.get("length_m", 100) / 1000
        total_cost = intervention["cost_per_km_inr"] * segment_length_km

        # ROI calculation
        cost_per_life_saved = (
            total_cost / lives_saved_per_year
            if lives_saved_per_year > 0 else 0
        )

        # ROI in years (how many years to break even
        # if each life saved = VALUE_OF_STATISTICAL_LIFE_INR)
        value_saved_per_year = lives_saved_per_year * VALUE_OF_STATISTICAL_LIFE_INR
        roi_years = (
            total_cost / value_saved_per_year
            if value_saved_per_year > 0 else 999
        )

        # Recommendation
        if risk_reduction_pct >= 25 and roi_years <= 3:
            recommendation = "HIGHLY RECOMMENDED"
        elif risk_reduction_pct >= 15 and roi_years <= 5:
            recommendation = "RECOMMENDED"
        elif risk_reduction_pct >= 10 and roi_years <= 10:
            recommendation = "CONSIDER"
        else:
            recommendation = "NOT RECOMMENDED"

        return {
            "scenario_type": "intervention",
            "segment_id": segment_id,
            "intervention_id": intervention_id,
            "intervention_name": intervention["name"],
            "intervention_description": intervention["description"],
            "base_risk": base_risk,
            "new_risk": new_risk,
            "risk_reduction": round(risk_reduction, 2),
            "risk_reduction_pct": round(risk_reduction_pct, 2),
            "lives_saved_per_year": round(lives_saved_per_year, 2),
            "segment_length_km": round(segment_length_km, 3),
            "cost_total_inr": round(total_cost, 2),
            "cost_per_km_inr": intervention["cost_per_km_inr"],
            "cost_per_life_saved_inr": round(cost_per_life_saved, 2),
            "roi_years": round(roi_years, 2),
            "value_saved_per_year_inr": round(value_saved_per_year, 2),
            "implementation_months": intervention["implementation_months"],
            "primary_benefit": intervention["primary_benefit"],
            "recommendation": recommendation,
        }

    # ─────────────────────────────────────────
    # MULTI-SCENARIO COMPARISON
    # ─────────────────────────────────────────

    def compare_interventions(self, segment_id: str,
                               intervention_ids: List[str]) -> dict:
        """
        Compare multiple interventions for a segment.

        Args:
            segment_id: Segment identifier
            intervention_ids: List of intervention IDs to compare

        Returns:
            Comparison result dict with sorted interventions
        """
        results = []

        for intervention_id in intervention_ids:
            try:
                result = self.simulate_intervention(
                    segment_id, intervention_id
                )
                results.append(result)
            except Exception as e:
                logger.warning(
                    f"Intervention {intervention_id} failed: {e}"
                )
                continue

        # Sort by risk reduction percentage (descending)
        results_sorted = sorted(
            results,
            key=lambda x: x["risk_reduction_pct"],
            reverse=True
        )

        # Best intervention
        best = results_sorted[0] if results_sorted else None

        return {
            "segment_id": segment_id,
            "interventions_compared": len(results),
            "results": results_sorted,
            "best_intervention": best["intervention_id"] if best else None,
            "best_risk_reduction_pct": best["risk_reduction_pct"] if best else 0,
        }

    def simulate_combined_scenario(self, segment_id: str,
                                     weather: Optional[str] = None,
                                     time_period: Optional[str] = None,
                                     traffic_level: Optional[str] = None) -> dict:
        """
        Simulate combined multi-factor scenario.

        Args:
            segment_id: Segment identifier
            weather: Weather condition (optional)
            time_period: Time period (optional)
            traffic_level: Traffic level (optional)

        Returns:
            Simulation result dict
        """
        base_input = self._create_base_input(segment_id)
        base_risk = self._predict_risk(base_input)

        modified_input = base_input.copy()

        # Apply modifications
        if weather:
            modified_input = self._modify_weather(modified_input, weather)
        if time_period:
            modified_input = self._modify_time(modified_input, time_period)
        if traffic_level:
            modified_input = self._modify_traffic(modified_input, traffic_level)

        new_risk = self._predict_risk(modified_input)

        risk_change = new_risk - base_risk
        risk_change_pct = (
            (risk_change / base_risk * 100) if base_risk > 0 else 0
        )

        return {
            "scenario_type": "combined",
            "segment_id": segment_id,
            "weather": weather or base_input["Weather_Conditions"],
            "time_period": time_period or "Day",
            "traffic_level": traffic_level or "Medium",
            "base_risk": base_risk,
            "new_risk": new_risk,
            "risk_change": round(risk_change, 2),
            "risk_change_pct": round(risk_change_pct, 2),
            "description": self._generate_scenario_description(
                weather, time_period, traffic_level
            ),
        }

    def _generate_scenario_description(self, weather, time_period,
                                         traffic_level) -> str:
        """Generate human-readable scenario description."""
        parts = []
        if weather:
            parts.append(f"{weather} weather")
        if time_period:
            parts.append(f"during {time_period.lower()}")
        if traffic_level:
            parts.append(f"with {traffic_level.lower()} traffic")

        if not parts:
            return "Base conditions"

        return "Scenario: " + ", ".join(parts)

    # ─────────────────────────────────────────
    # BATCH ANALYSIS
    # ─────────────────────────────────────────

    def analyze_top_segments(self, intervention_id: str,
                              n: int = 10) -> dict:
        """
        Analyze intervention impact on top N dangerous segments.

        Args:
            intervention_id: Intervention to analyze
            n: Number of top segments to analyze

        Returns:
            Batch analysis result dict
        """
        # Get top dangerous segments
        sorted_segments = sorted(
            self.segment_risks.values(),
            key=lambda x: x["composite_risk"],
            reverse=True
        )[:n]

        results = []
        total_risk_reduction = 0
        total_lives_saved = 0
        total_cost = 0

        for segment in sorted_segments:
            segment_id = segment["segment_id"]

            try:
                result = self.simulate_intervention(
                    segment_id, intervention_id
                )
                results.append(result)

                total_risk_reduction += result["risk_reduction_pct"]
                total_lives_saved += result["lives_saved_per_year"]
                total_cost += result["cost_total_inr"]

            except Exception as e:
                logger.warning(
                    f"Analysis failed for {segment_id}: {e}"
                )
                continue

        avg_risk_reduction = (
            total_risk_reduction / len(results) if results else 0
        )

        total_value_saved_per_year = total_lives_saved * VALUE_OF_STATISTICAL_LIFE_INR
        overall_roi_years = (
            total_cost / total_value_saved_per_year
            if total_value_saved_per_year > 0 else 999
        )

        return {
            "intervention_id": intervention_id,
            "intervention_name": INTERVENTIONS[intervention_id]["name"],
            "segments_analyzed": len(results),
            "total_risk_reduction_pct": round(total_risk_reduction, 2),
            "avg_risk_reduction_pct": round(avg_risk_reduction, 2),
            "total_lives_saved_per_year": round(total_lives_saved, 2),
            "total_cost_inr": round(total_cost, 2),
            "total_value_saved_per_year_inr": round(total_value_saved_per_year, 2),
            "overall_roi_years": round(overall_roi_years, 2),
            "cost_per_life_saved_inr": round(
                total_cost / total_lives_saved if total_lives_saved > 0 else 0,
                2
            ),
            "segments": results,
        }

    # ─────────────────────────────────────────
    # SAVE / LOAD
    # ─────────────────────────────────────────

    def save_scenarios(self, scenarios: dict = None):
        """Save simulated scenarios to JSON."""
        if scenarios is None:
            scenarios = self.simulated_scenarios

        with open(self.scenarios_path, "w") as f:
            json.dump(scenarios, f, indent=2)

        logger.info(f"Scenarios saved: {self.scenarios_path}")

    def load_scenarios(self) -> dict:
        """Load simulated scenarios from JSON."""
        if not os.path.exists(self.scenarios_path):
            return {}

        with open(self.scenarios_path, "r") as f:
            scenarios = json.load(f)

        self.simulated_scenarios = scenarios
        return scenarios
