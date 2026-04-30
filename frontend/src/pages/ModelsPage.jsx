import React, { useState, useEffect } from 'react';
import { FiAward, FiPercent, FiClock } from 'react-icons/fi';
import { api } from '../api/apiClient';
import KPICard from '../components/common/KPICard';
import ChartTile from '../components/common/ChartTile';
import DataTable from '../components/common/DataTable';
import ConfusionMatrix from '../components/common/ConfusionMatrix';
import ModelComparisonBar from '../components/charts/ModelComparisonBar';
import BoxPlotChart from '../components/charts/BoxPlotChart';
import ROCCurve from '../components/charts/ROCCurve';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { formatMetric, formatTime } from '../utils/formatters';

const ModelsPage = () => {
  const [loading, setLoading] = useState(true);
  const [comparison, setComparison] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);
  const [cmData, setCmData] = useState(null);
  const [rocDataAll, setRocDataAll] = useState({});

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const res = await api.modelComparison();
        setComparison(res.data);
        if (res.data.models?.length) {
          setSelectedModel(res.data.best_model || res.data.models[0].model_name);
          const rocMap = {};
          res.data.models.forEach((m) => {
            if (m.roc_data) rocMap[m.model_name] = m.roc_data;
          });
          setRocDataAll(rocMap);
        }
      } catch (err) {
        console.error('Failed to fetch model comparison:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  useEffect(() => {
    if (!selectedModel) return;
    const fetchCM = async () => {
      try {
        const res = await api.modelConfusionMatrix(selectedModel);
        setCmData(res.data);
      } catch (err) {
        console.error('Failed to fetch confusion matrix:', err);
      }
    };
    fetchCM();
  }, [selectedModel]);

  if (loading) return <LoadingSpinner text="Loading model results..." />;
  if (!comparison) return <p className="text-pbi-muted">No model data available</p>;

  const models = comparison.models || [];
  const best = models.find((m) => m.model_name === comparison.best_model) || models[0];

  const tableColumns = [
    { key: 'model_name', label: 'Model', render: (v) => <span className="font-semibold text-white">{v}</span> },
    { key: 'accuracy', label: 'Accuracy', render: (v) => formatMetric(v) },
    { key: 'precision_weighted', label: 'Precision', render: (v) => formatMetric(v) },
    { key: 'recall_weighted', label: 'Recall', render: (v) => formatMetric(v) },
    { key: 'f1_weighted', label: 'F1(W)', render: (v) => formatMetric(v) },
    { key: 'f1_macro', label: 'F1(M)', render: (v) => formatMetric(v) },
    { key: 'roc_auc', label: 'AUC', render: (v) => formatMetric(v) },
    { key: 'cohens_kappa', label: 'Kappa', render: (v) => formatMetric(v) },
    { key: 'mcc', label: 'MCC', render: (v) => formatMetric(v) },
    { key: 'log_loss', label: 'LogLoss', render: (v) => formatMetric(v) },
    { key: 'training_time', label: 'Time', render: (v) => formatTime(v) },
  ];

  return (
    <div className="space-y-6">
      {/* KPIs */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <KPICard icon={<FiAward />} value={best.model_name} label="Best Model" subtext={`F1: ${formatMetric(best.f1_weighted)}`} color="#10B981" />
        <KPICard icon={<FiPercent />} value={`${(best.accuracy * 100).toFixed(1)}%`} label="Highest Accuracy" subtext={best.model_name} color="#2563EB" delay={0.1} />
        <KPICard icon={<FiClock />} value={models.length} label="Models Trained" subtext="Across datasets" color="#8B5CF6" delay={0.2} />
      </div>

      {/* Comparison Table */}
      <ChartTile title="Model Comparison Table" subtitle="Click column headers to sort">
        <DataTable columns={tableColumns} data={models} pageSize={10} highlightBest sortable />
      </ChartTile>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <ChartTile title="Accuracy & F1 Score Comparison">
          <ModelComparisonBar models={models} />
        </ChartTile>
        <ChartTile title="Cross-Validation Scores">
          <BoxPlotChart models={models} />
        </ChartTile>
      </div>

      {/* Confusion Matrix */}
      <ChartTile title="Confusion Matrix">
        <div className="flex gap-2 mb-4 flex-wrap">
          {models.map((m) => (
            <button key={m.model_name} onClick={() => setSelectedModel(m.model_name)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all
                         ${selectedModel === m.model_name ? 'bg-pbi-blue text-white' : 'bg-pbi-bg2 text-pbi-muted hover:text-white border border-pbi-border'}`}>
              {m.model_name}
            </button>
          ))}
        </div>
        {cmData && <ConfusionMatrix matrix={cmData.matrix} normalizedMatrix={cmData.normalized_matrix} labels={cmData.labels} />}
      </ChartTile>

      {/* ROC Curves */}
      <ChartTile title="ROC Curves (One-vs-Rest)">
        <ROCCurve rocDataByModel={rocDataAll} />
      </ChartTile>
    </div>
  );
};

export default ModelsPage;
