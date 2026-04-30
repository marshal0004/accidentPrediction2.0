import React from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from 'recharts';
import { CHART_COLORS } from '../../utils/constants';

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-pbi-card border border-pbi-border rounded-lg p-3 shadow-xl">
      <p className="text-sm font-semibold text-white mb-1">{label}</p>
      {payload.map((p, i) => (
        <p key={i} className="text-xs" style={{ color: p.color }}>{p.name}: {p.value}</p>
      ))}
    </div>
  );
};

const LineChartComponent = ({ data }) => {
  if (!data || !data.labels || !data.datasets) return <p className="text-pbi-muted text-sm">No data</p>;

  const chartData = data.labels.map((label, i) => {
    const point = { name: label };
    data.datasets.forEach((ds) => {
      point[ds.label] = ds.data[i] || 0;
    });
    return point;
  });

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 40 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#2D374830" />
        <XAxis dataKey="name" tick={{ fill: '#8899A6', fontSize: 8 }} angle={-45} textAnchor="end" tickLine={false} interval="preserveStartEnd" />
        <YAxis tick={{ fill: '#8899A6', fontSize: 11 }} tickLine={false} axisLine={false} />
        <Tooltip content={<CustomTooltip />} />
        <Legend formatter={(v) => <span className="text-xs text-pbi-text2">{v}</span>} />
        {data.datasets.map((ds, i) => (
          <Line key={ds.label} type="monotone" dataKey={ds.label} stroke={ds.color || CHART_COLORS[i]} strokeWidth={2} dot={false} animationDuration={1500} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
};

export default LineChartComponent;
