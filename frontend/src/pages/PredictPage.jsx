import React, { useState } from 'react';
import { api } from '../api/apiClient';
import ChartTile from '../components/common/ChartTile';
import PredictionForm from '../components/prediction/PredictionForm';
import PredictionResult from '../components/prediction/PredictionResult';
import BatchUpload from '../components/prediction/BatchUpload';
import toast from 'react-hot-toast';

const PredictPage = () => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handlePredict = async (formData) => {
    setLoading(true);
    setResult(null);
    try {
      const res = await api.predict(formData);
      setResult(res.data);
      toast.success(`Predicted: ${res.data.prediction} (${(res.data.confidence * 100).toFixed(1)}% confidence)`);
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Prediction Form */}
        <ChartTile title="Enter Accident Details" subtitle="Fill in the details to predict severity">
          <PredictionForm onSubmit={handlePredict} loading={loading} />
        </ChartTile>

        {/* Result */}
        <div>
          {result ? (
            <PredictionResult result={result} />
          ) : (
            <div className="glass-card-static p-8 flex flex-col items-center justify-center h-full text-center">
              <div className="text-5xl mb-4">🎯</div>
              <h3 className="text-lg font-semibold text-white mb-2">Ready to Predict</h3>
              <p className="text-sm text-pbi-muted max-w-sm">
                Fill in the accident details on the left and click "Predict Severity" to get AI-powered prediction with confidence scores.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Batch Upload */}
      <ChartTile title="Batch Prediction" subtitle="Upload a CSV file for bulk predictions">
        <BatchUpload />
      </ChartTile>
    </div>
  );
};

export default PredictPage;
