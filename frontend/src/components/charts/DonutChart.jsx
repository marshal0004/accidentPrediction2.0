import React from 'react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { SEVERITY_COLORS, CHART_COLORS } from '../../utils/constants';

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const { name, value, percent } = payload[0].payload;
  return (
    <div className="bg-pbi-card border border-pbi-border rounded-lg p-3 shadow-xl">
      <p className="text-sm font-semibold text-white">{name}</p>
      <p className="text-xs text-pbi-text2">{value.toLocaleString()} incidents</p>
      <p className="text-xs text-pbi-blue">{(percent * 100).toFixed(1)}%</p>
    </div>
  );
};

const DonutChart = ({ data }) => {
  if (!data || !data.labels) return null;

  const total = data.values.reduce((a, b) => a + b, 0);
  const chartData = data.labels.map((label, i) => ({
    name: label,
    value: data.values[i],
    percent: data.values[i] / total,
  }));

  const colors = data.colors || data.labels.map(l => SEVERITY_COLORS[l] || CHART_COLORS[chartData.indexOf(chartData.find(d => d.name === l)) % CHART_COLORS.length]);

  return (
    <ResponsiveContainer width="100%" height={280}>
      <PieChart>
        <Pie
          data={chartData}
          cx="50%"
          cy="50%"
          innerRadius={60}
          outerRadius={100}
          paddingAngle={3}
          dataKey="value"
          animationBegin={0}
          animationDuration={1200}
          stroke="none"
        >
          {chartData.map((_, idx) => (
            <Cell key={idx} fill={colors[idx] || CHART_COLORS[idx % CHART_COLORS.length]} />
          ))}
        </Pie>
        <Tooltip content={<CustomTooltip />} />
        <Legend
          verticalAlign="bottom"
          height={36}
          formatter={(value) => <span className="text-xs text-pbi-text2">{value}</span>}
        />
        <text x="50%" y="48%" textAnchor="middle" className="fill-white text-2xl font-bold">
          {total.toLocaleString()}
        </text>
        <text x="50%" y="56%" textAnchor="middle" className="fill-pbi-muted text-xs">
          Total
        </text>
      </PieChart>
    </ResponsiveContainer>
  );
};

export default DonutChart;
