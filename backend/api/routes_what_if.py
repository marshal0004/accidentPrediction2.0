# backend/api/routes_what_if.py

from fastapi import APIRouter, HTTPException, Query, Body
from typing import List, Optional
from pydantic import BaseModel, Field
import logging

from config import INTERVENTIONS
from api.routes_digital_twin import get_twin

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────
router = APIRouter(prefix="/api/what-if", tags=["What-If Analysis"])

# ─────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────

class InterventionAnalysisRequest(BaseModel):
    intervention_id: str = Field(..., description="Intervention identifier")


class CompareInterventionsRequest(BaseModel):
    intervention_ids: List[str] = Field(..., description="List of intervention IDs to compare")


class BatchAnalysisRequest(BaseModel):
    segment_ids: List[str] = Field(..., description="List of segment IDs")
    intervention_id: str = Field(..., description="Intervention to analyze")


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@router.get("/interventions")
async def list_interventions():
    """
    List all available safety interventions.
    
    Returns intervention details including:
    - Name and description
    - Cost per km
    - Expected risk reduction
    - Implementation time
    """
    interventions = []
    
    for key, config in INTERVENTIONS.items():
        interventions.append({
            "id": key,
            "name": config["name"],
            "description": config["description"],
            "cost_per_km_inr": config["cost_per_km_inr"],
            "expected_risk_reduction_min": config["expected_risk_reduction_min"],
            "expected_risk_reduction_max": config["expected_risk_reduction_max"],
            "implementation_months": config["implementation_months"],
            "primary_benefit": config["primary_benefit"],
        })
    
    return {
        "interventions": interventions,
        "total_interventions": len(interventions),
    }


@router.post("/{city_key}/segment/{segment_id}/analyze")
async def analyze_intervention(
    city_key: str,
    segment_id: str,
    request: InterventionAnalysisRequest
):
    """
    Analyze impact of a safety intervention on a segment.
    
    Returns:
    - Risk reduction percentage
    - Lives saved per year
    - Total cost
    - ROI analysis
    - Recommendation
    """
    twin = get_twin(city_key)
    
    if not twin.scenario_simulator:
        raise HTTPException(
            status_code=503,
            detail="Scenario simulator not available"
        )
    
    intervention_id = request.intervention_id
    
    if intervention_id not in INTERVENTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown intervention: {intervention_id}. "
                   f"Available: {list(INTERVENTIONS.keys())}"
        )
    
    try:
        result = twin.scenario_simulator.simulate_intervention(
            segment_id, intervention_id
        )
        
        return {
            "city": city_key,
            "segment_id": segment_id,
            "intervention": intervention_id,
            "analysis": result,
        }
        
    except Exception as e:
        logger.error(f"Intervention analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post("/{city_key}/segment/{segment_id}/compare")
async def compare_interventions(
    city_key: str,
    segment_id: str,
    request: CompareInterventionsRequest
):
    """
    Compare multiple interventions for a segment.
    
    Returns interventions sorted by effectiveness (risk reduction).
    """
    twin = get_twin(city_key)
    
    if not twin.scenario_simulator:
        raise HTTPException(
            status_code=503,
            detail="Scenario simulator not available"
        )
    
    intervention_ids = request.intervention_ids
    
    # Validate interventions
    for intervention_id in intervention_ids:
        if intervention_id not in INTERVENTIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown intervention: {intervention_id}"
            )
    
    try:
        result = twin.scenario_simulator.compare_interventions(
            segment_id, intervention_ids
        )
        
        return {
            "city": city_key,
            "segment_id": segment_id,
            "comparison": result,
        }
        
    except Exception as e:
        logger.error(f"Intervention comparison failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {str(e)}"
        )


@router.post("/{city_key}/batch-analyze")
async def batch_analyze_intervention(
    city_key: str,
    request: BatchAnalysisRequest
):
    """
    Analyze intervention impact across multiple segments.
    
    Useful for:
    - Citywide intervention planning
    - Budget allocation
    - Prioritizing high-impact locations
    """
    twin = get_twin(city_key)
    
    if not twin.scenario_simulator:
        raise HTTPException(
            status_code=503,
            detail="Scenario simulator not available"
        )
    
    segment_ids = request.segment_ids
    intervention_id = request.intervention_id
    
    if intervention_id not in INTERVENTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown intervention: {intervention_id}"
        )
    
    if len(segment_ids) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 segments allowed per batch analysis"
        )
    
    try:
        results = []
        total_risk_reduction = 0
        total_lives_saved = 0
        total_cost = 0
        
        for segment_id in segment_ids:
            try:
                result = twin.scenario_simulator.simulate_intervention(
                    segment_id, intervention_id
                )
                results.append(result)
                
                total_risk_reduction += result.get("risk_reduction_pct", 0)
                total_lives_saved += result.get("lives_saved_per_year", 0)
                total_cost += result.get("cost_total_inr", 0)
                
            except Exception as e:
                logger.warning(f"Segment {segment_id} failed: {e}")
                continue
        
        if not results:
            raise HTTPException(
                status_code=500,
                detail="All segments failed analysis"
            )
        
        avg_risk_reduction = total_risk_reduction / len(results)
        
        # Calculate overall ROI
        from config import VALUE_OF_STATISTICAL_LIFE_INR
        total_value_saved = total_lives_saved * VALUE_OF_STATISTICAL_LIFE_INR
        overall_roi_years = (
            total_cost / total_value_saved
            if total_value_saved > 0 else 999
        )
        
        return {
            "city": city_key,
            "intervention_id": intervention_id,
            "intervention_name": INTERVENTIONS[intervention_id]["name"],
            "segments_analyzed": len(results),
            "segments_failed": len(segment_ids) - len(results),
            "total_risk_reduction_pct": round(total_risk_reduction, 2),
            "avg_risk_reduction_pct": round(avg_risk_reduction, 2),
            "total_lives_saved_per_year": round(total_lives_saved, 2),
            "total_cost_inr": round(total_cost, 2),
            "total_value_saved_per_year_inr": round(total_value_saved, 2),
            "overall_roi_years": round(overall_roi_years, 2),
            "cost_per_life_saved_inr": round(
                total_cost / total_lives_saved if total_lives_saved > 0 else 0,
                2
            ),
            "results": results,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch analysis failed: {str(e)}"
        )


@router.get("/{city_key}/recommendations")
async def get_intervention_recommendations(
    city_key: str,
    budget: float = Query(..., description="Available budget in INR"),
    max_segments: int = Query(10, ge=1, le=50, description="Maximum segments to analyze")
):
    """
    Get optimized intervention recommendations within budget.
    
    Returns prioritized list of segment-intervention pairs
    that maximize lives saved within budget constraints.
    """
    twin = get_twin(city_key)
    
    if not twin.scenario_simulator:
        raise HTTPException(
            status_code=503,
            detail="Scenario simulator not available"
        )
    
    try:
        # Get top dangerous segments
        top_segments = twin.get_top_dangerous_segments(n=max_segments, min_risk=60)
        
        recommendations = []
        total_cost = 0
        total_lives_saved = 0
        
        # For each segment, find best intervention
        for segment in top_segments:
            segment_id = segment["segment_id"]
            
            # Compare all interventions
            best_intervention = None
            best_lives_saved = 0
            best_cost = 0
            best_result = None
            
            for intervention_id in INTERVENTIONS.keys():
                try:
                    result = twin.scenario_simulator.simulate_intervention(
                        segment_id, intervention_id
                    )
                    
                    cost = result["cost_total_inr"]
                    lives_saved = result["lives_saved_per_year"]
                    
                    # Check if within remaining budget
                    if total_cost + cost <= budget:
                        # Choose intervention with most lives saved
                        if lives_saved > best_lives_saved:
                            best_lives_saved = lives_saved
                            best_intervention = intervention_id
                            best_cost = cost
                            best_result = result
                            
                except Exception as e:
                    logger.debug(f"Intervention {intervention_id} failed: {e}")
                    continue
            
            # Add best intervention to recommendations
            if best_intervention and best_result:
                recommendations.append({
                    "segment_id": segment_id,
                    "road_name": segment.get("road_name", "Unknown"),
                    "current_risk": segment.get("composite_risk", 0),
                    "intervention_id": best_intervention,
                    "intervention_name": INTERVENTIONS[best_intervention]["name"],
                    "lives_saved_per_year": best_lives_saved,
                    "cost_inr": best_cost,
                    "risk_reduction_pct": best_result.get("risk_reduction_pct", 0),
                    "roi_years": best_result.get("roi_years", 0),
                    "recommendation": best_result.get("recommendation", ""),
                })
                
                total_cost += best_cost
                total_lives_saved += best_lives_saved
        
        # Sort by lives saved (descending)
        recommendations.sort(
            key=lambda x: x["lives_saved_per_year"],
            reverse=True
        )
        
        return {
            "city": city_key,
            "budget_available_inr": budget,
            "budget_used_inr": round(total_cost, 2),
            "budget_remaining_inr": round(budget - total_cost, 2),
            "total_lives_saved_per_year": round(total_lives_saved, 2),
            "segments_recommended": len(recommendations),
            "recommendations": recommendations,
        }
        
    except Exception as e:
        logger.error(f"Recommendations generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Recommendations failed: {str(e)}"
        )
