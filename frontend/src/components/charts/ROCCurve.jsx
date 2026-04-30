import React from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend, ReferenceLine } from 'recharts';
import { MODEL_COLORS, CHART_COLORS } from '../../utils/constants';

const ROCCurve = ({ rocDataByModel }) => {
  if (!rocDataByModel || !Object.keys(rocDataByModel).length) return <p className="text-pbi-muted text-sm">No ROC data</p>;

  const allClasses = new Set();
  Object.values(rocDataByModel).forEach((classes) => {
    Object.keys(classes).forEach((cls) => allClasses.add(cls));
  });
  const classNames = Array.from(allClasses);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {classNames.map((cls) => {
        const lines = [];
        Object.entries(rocDataByModel).forEach(([modelName, classes]) => {
          if (classes[cls]) {
            const points = classes[cls].fpr.map((fpr, i) => ({
              fpr,
              [modelName]: classes[cls].tpr[i],
            }));
            lines.push({ modelName, points, auc: classes[cls].auc });
          }
        });

        const mergedData = [];
        const allFpr = new Set();
        lines.forEach(({ points }) => points.forEach((p) => allFpr.add(p.fpr)));
        const sortedFpr = Array.from(allFpr).sort((a, b) => a - b);

        sortedFpr.forEach((fpr) => {
          const point = { fpr };
          lines.forEach(({ modelName, points }) => {
            const match = points.find((p) => p.fpr === fpr);
            if (match) point[modelName] = match[modelName];
          });
          mergedData.push(point);
        });

        return (
          <div key={cls} className="glass-card-static p-4">
            <h4 className="text-xs font-semibold text-pbi-text2 mb-3">{cls} (One-vs-Rest)</h4>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={mergedData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2D374830" />
                <XAxis dataKey="fpr" tick={{ fill: '#8899A6', fontSize: 9 }} />
                <YAxis tick={{ fill: '#8899A6', fontSize: 9 }} domain={[0, 1]} />
                <Tooltip contentStyle={{ backgroundColor: '#1F2940', border: '1px solid #2D3748', borderRadius: 8 }} />
                <ReferenceLine x={0} y={0} stroke="#5C6B7A" strokeDasharray="5 5" segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} />
                {lines.map(({ modelName, auc }, i) => (
                  <Line key={modelName} type="monotone" dataKey={modelName} stroke={MODEL_COLORS[modelName] || CHART_COLORS[i]} strokeWidth={2} dot={false}
                    name={`${modelName} (${auc.toFixed(3)})`} />
                ))}
                <Legend formatter={(v) => <span className="text-[10px] text-pbi-text2">{v}</span>} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        );
      })}
    </div>
  );
};

export default ROCCurve;
