import React, { useState, useEffect } from 'react';
import { api } from '../api/apiClient';
import ChartTile from '../components/common/ChartTile';
import HorizontalBarChart from '../components/charts/HorizontalBarChart';
import CorrelationHeatmap from '../components/charts/CorrelationHeatmap';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { CHART_COLORS } from '../utils/constants';

const FeaturesPage = () => {
  const [loading, setLoading] = useState(true);
  const [shapData, setShapData] = useState(null);
  const [allShap, setAllShap] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);
  const [summaryPlot, setSummaryPlot] = useState(null);
  const [correlationData, setCorrelationData] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const [shapRes, allShapRes, corrRes] = await Promise.allSettled([
          api.shapFeatureImportance(),
          api.shapAllModels(),
          api.edaChart('correlation_matrix'),
        ]);

        if (shapRes.status === 'fulfilled') {
          setShapData(shapRes.value.data);
          setSelectedModel(shapRes.value.data.best_model || null);
        }
        if (allShapRes.status === 'fulfilled') setAllShap(allShapRes.value.data);
        if (corrRes.status === 'fulfilled') setCorrelationData(corrRes.value.data);
      } catch (err) {
        console.error('Failed to fetch SHAP data:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  useEffect(() => {
    if (!selectedModel) return;
    const fetchPlot = async () => {
      try {
        const res = await api.shapSummaryPlot(selectedModel);
        setSummaryPlot(res.data.image_base64);
      } catch {
        setSummaryPlot(null);
      }
    };
    fetchPlot();
  }, [selectedModel]);

  const handleModelSelect = async (modelName) => {
    setSelectedModel(modelName);
    try {
      const res = await api.shapFeatureImportance(modelName);
      setShapData(res.data);
    } catch (err) {
      console.error('Failed to fetch SHAP for model:', err);
    }
  };

  if (loading) return <LoadingSpinner text="Loading SHAP analysis..." />;

  const features = shapData?.features || [];
  const shapBarData = {
    labels: features.map((f) => f.name),
    values: features.map((f) => f.importance),
  };

  const modelNames = allShap?.models ? Object.keys(allShap.models) : [];

  return (
    <div className="space-y-6">
      {/* Model Selector */}
      {modelNames.length > 0 && (
        <div className="glass-card-static p-4">
          <h3 className="text-xs font-semibold text-pbi-muted uppercase tracking-wider mb-3">Select Model for SHAP Analysis</h3>
          <div className="flex gap-2 flex-wrap">
            {modelNames.map((name, i) => (
              <button key={name} onClick={() => handleModelSelect(name)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all
                           ${selectedModel === name ? 'text-white shadow-lg' : 'bg-pbi-bg2 text-pbi-muted hover:text-white border border-pbi-border'}`}
                style={selectedModel === name ? { backgroundColor: CHART_COLORS[i % CHART_COLORS.length] } : {}}>
                {name}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* SHAP Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <ChartTile title={`SHAP Feature Importance — ${selectedModel || 'Best Model'}`} subtitle="Top 15 features by mean |SHAP value|">
          <HorizontalBarChart data={shapBarData} />
        </ChartTile>

        <ChartTile title="SHAP Summary Plot (Beeswarm)">
          {summaryPlot ? (
            <img src={summaryPlot} alt="SHAP Summary" className="w-full h-auto rounded-lg" />
          ) : (
            <p className="text-pbi-muted text-sm text-center py-8">Select a model to view SHAP plot</p>
          )}
        </ChartTile>
      </div>

      {/* Feature Table */}
      {features.length > 0 && (
        <ChartTile title="Feature Importance Table">
          <div className="space-y-2">
            {features.map((f, i) => (
              <div key={f.name} className="flex items-center gap-3">
                <span className="text-xs text-pbi-muted w-6 text-right">{i + 1}</span>
                <div className="flex-1">
                  <div className="flex justify-between mb-1">
                    <span className="text-sm text-white">{f.name}</span>
                    <span className="text-xs text-pbi-blue font-semibold">{f.importance.toFixed(6)}</span>
                  </div>
                  <div className="h-1.5 bg-pbi-bg rounded-full overflow-hidden">
                    <div className="h-full bg-pbi-blue rounded-full transition-all duration-700"
                      style={{ width: `${(f.importance / (features[0]?.importance || 1)) * 100}%` }} />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </ChartTile>
      )}

      {/* Correlation Heatmap */}
      <ChartTile title="Feature Correlation Matrix">
        <CorrelationHeatmap data={correlationData} />
      </ChartTile>
    </div>
  );
};

export default FeaturesPage;
