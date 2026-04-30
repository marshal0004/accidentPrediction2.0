import React, { useState, useEffect } from 'react';
import { FiAlertTriangle, FiXCircle, FiCheckCircle, FiTrendingUp } from 'react-icons/fi';
import { api } from '../api/apiClient';
import KPICard from '../components/common/KPICard';
import ChartTile from '../components/common/ChartTile';
import FilterBar from '../components/common/FilterBar';
import DonutChart from '../components/charts/DonutChart';
import AreaTimeChart from '../components/charts/AreaTimeChart';
import HorizontalBarChart from '../components/charts/HorizontalBarChart';
import GroupedBarChart from '../components/charts/GroupedBarChart';
import LineChartComponent from '../components/charts/LineChartComponent';
import { SkeletonCard, SkeletonChart } from '../components/common/LoadingSpinner';
import { formatNumber } from '../utils/formatters';

const OverviewPage = () => {
  const [loading, setLoading] = useState(true);
  const [summary, setSummary] = useState(null);
  const [charts, setCharts] = useState({});
  const [bestModel, setBestModel] = useState(null);
  const [shapTop, setShapTop] = useState(null);
  const [datasetKey, setDatasetKey] = useState('primary');
  const [filters, setFilters] = useState({});

  useEffect(() => {
    const fetchAll = async () => {
      setLoading(true);
      try {
        const [summaryRes, bestRes, shapRes] = await Promise.allSettled([
          api.edaSummary(),
          api.bestModel(),
          api.shapFeatureImportance(),
        ]);

        if (summaryRes.status === 'fulfilled') setSummary(summaryRes.value.data);
        if (bestRes.status === 'fulfilled') setBestModel(bestRes.value.data);
        if (shapRes.status === 'fulfilled') setShapTop(shapRes.value.data);

        const chartNames = ['class_distribution', 'accidents_by_hour', 'severity_by_cause', 'accidents_by_weather', 'monthly_trend', 'accidents_by_vehicle'];
        const chartResults = await Promise.allSettled(
          chartNames.map((name) => api.edaChart(name, datasetKey))
        );

        const chartsObj = {};
        chartNames.forEach((name, i) => {
          if (chartResults[i].status === 'fulfilled') {
            chartsObj[name] = chartResults[i].value.data;
          }
        });
        setCharts(chartsObj);
      } catch (err) {
        console.error('Failed to fetch overview data:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchAll();
  }, [datasetKey]);

  const ds = summary?.[datasetKey] || summary?.primary || {};
  const totalRecords = ds.total_records || 0;
  const classDist = ds.class_distribution || {};
  const fatalCount = classDist['Fatal'] || classDist['fatal'] || Object.values(classDist)[0] || 0;
  const fatalPct = totalRecords > 0 ? ((fatalCount / totalRecords) * 100).toFixed(1) : 0;

  const filterConfig = [
    { key: 'dataset', label: 'Dataset', options: ['primary', 'secondary'] },
  ];

  const handleFilterChange = (key, value) => {
    if (key === 'dataset' && value) setDatasetKey(value);
    setFilters((prev) => ({ ...prev, [key]: value }));
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
          {[...Array(4)].map((_, i) => <SkeletonCard key={i} />)}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          {[...Array(4)].map((_, i) => <SkeletonChart key={i} />)}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* KPI Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
        <KPICard icon={<FiAlertTriangle />} value={formatNumber(totalRecords)} label="Total Accidents" subtext={ds.name || ''} color="#F97316" delay={0} />
        <KPICard icon={<FiXCircle />} value={formatNumber(fatalCount)} label="Fatal Accidents" subtext={`${fatalPct}% of total`} color="#EF4444" delay={0.1} />
        <KPICard icon={<FiCheckCircle />} value={bestModel ? `${(bestModel.best_value * 100).toFixed(1)}%` : '—'} label="Best Model F1" subtext={bestModel?.best_model || ''} color="#10B981" delay={0.2} />
        <KPICard icon={<FiTrendingUp />} value={shapTop?.features?.[0]?.name || '—'} label="#1 Risk Factor" subtext={shapTop?.features?.[0] ? `SHAP: ${shapTop.features[0].importance.toFixed(4)}` : ''} color="#8B5CF6" delay={0.3} />
      </div>

      {/* Filters */}
      <FilterBar filters={filterConfig} values={filters} onChange={handleFilterChange} onReset={() => { setFilters({}); setDatasetKey('primary'); }} />

      {/* Main Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <ChartTile title="Severity Distribution" delay={0.1}>
          <DonutChart data={charts.class_distribution} />
        </ChartTile>
        <ChartTile title="Accidents by Time of Day" delay={0.2}>
          <AreaTimeChart data={charts.accidents_by_hour} />
        </ChartTile>
        <ChartTile title="Top Accident Causes by Severity" delay={0.3}>
          <HorizontalBarChart data={charts.severity_by_cause} />
        </ChartTile>
        <ChartTile title="Weather Impact on Severity" delay={0.4}>
          <GroupedBarChart data={charts.accidents_by_weather} />
        </ChartTile>
      </div>

      {/* Bottom Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <ChartTile title="Monthly Accident Trend" delay={0.5}>
          <LineChartComponent data={charts.monthly_trend} />
        </ChartTile>
        <ChartTile title="Vehicle Type Distribution" delay={0.6}>
          <DonutChart data={charts.accidents_by_vehicle} />
        </ChartTile>
      </div>
    </div>
  );
};

export default OverviewPage;
