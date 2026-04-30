import React, { useState, useEffect } from 'react';
import { FiSend } from 'react-icons/fi';
import { api } from '../../api/apiClient';

const PredictionForm = ({ onSubmit, loading }) => {
  const [filterOptions, setFilterOptions] = useState({});
  const [formData, setFormData] = useState({
    Day_of_Week: 'Monday',
    Time_of_Accident: '12:00',
    Accident_Location: 'Urban',
    Nature_of_Accident: 'Head-on Collision',
    Causes_D1: 'Overspeeding',
    Road_Condition: 'Straight',
    Weather_Conditions: 'Clear',
    Intersection_Type: 'None',
    Vehicle_Type_V1: 'Car',
    Vehicle_Type_V2: 'Truck',
    Number_of_Vehicles: 2,
    model_name: 'XGBoost',
  });

  useEffect(() => {
    const fetchOptions = async () => {
      try {
        const res = await api.filterOptions();
        setFilterOptions(res.data);
      } catch (err) {
        console.error('Failed to fetch filter options:', err);
      }
    };
    fetchOptions();
  }, []);

  const handleChange = (key, value) => {
    setFormData((prev) => ({ ...prev, [key]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(formData);
  };

  const fields = [
    { key: 'Day_of_Week', label: 'Day of Week', type: 'select', options: filterOptions.Day_of_Week || ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'] },
    { key: 'Time_of_Accident', label: 'Time of Accident', type: 'time' },
    { key: 'Accident_Location', label: 'Location Type', type: 'select', options: filterOptions.Accident_Location || ['Urban', 'Rural'] },
    { key: 'Nature_of_Accident', label: 'Nature of Accident', type: 'select', options: filterOptions.Nature_of_Accident || ['Head-on Collision','Rear-end Collision','Side Collision','Hit and Run','Overturning','Pedestrian Knock Down','Other'] },
    { key: 'Causes_D1', label: 'Primary Cause', type: 'select', options: filterOptions.Causes_D1 || filterOptions.Causes || ['Overspeeding','Drunk Driving','Wrong Side Driving','Distracted Driving','Other'] },
    { key: 'Road_Condition', label: 'Road Condition', type: 'select', options: filterOptions.Road_Conditions || ['Straight','Curve','Bridge','Intersection','Other'] },
    { key: 'Weather_Conditions', label: 'Weather', type: 'select', options: filterOptions.Weather_Conditions || ['Clear','Rainy','Foggy','Cloudy','Other'] },
    { key: 'Intersection_Type', label: 'Intersection Type', type: 'select', options: filterOptions.Intersection_Types || ['None','T-Junction','Y-Junction','Four-way','Roundabout'] },
    { key: 'Vehicle_Type_V1', label: 'Vehicle 1 Type', type: 'select', options: filterOptions.Vehicle_Types || ['Car','Truck','Bus','Two Wheeler','Auto Rickshaw'] },
    { key: 'Vehicle_Type_V2', label: 'Vehicle 2 Type', type: 'select', options: filterOptions.Vehicle_Types || ['Car','Truck','Bus','Two Wheeler','None'] },
    { key: 'Number_of_Vehicles', label: 'Number of Vehicles', type: 'number', min: 1, max: 10 },
    { key: 'model_name', label: 'ML Model', type: 'select', options: filterOptions.Models || ['RandomForest','XGBoost','GradientBoosting','SVM','LogisticRegression'] },
  ];

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {fields.map((field) => (
          <div key={field.key} className="flex flex-col gap-1.5">
            <label className="text-xs font-medium text-pbi-muted uppercase tracking-wider">
              {field.label}
            </label>
            {field.type === 'select' ? (
              <select
                value={formData[field.key]}
                onChange={(e) => handleChange(field.key, e.target.value)}
                className="pbi-select"
              >
                {field.options.map((opt) => (
                  <option key={opt} value={opt}>{opt}</option>
                ))}
              </select>
            ) : field.type === 'time' ? (
              <input
                type="time"
                value={formData[field.key]}
                onChange={(e) => handleChange(field.key, e.target.value)}
                className="pbi-input"
              />
            ) : (
              <input
                type="number"
                value={formData[field.key]}
                onChange={(e) => handleChange(field.key, parseInt(e.target.value) || 1)}
                min={field.min}
                max={field.max}
                className="pbi-input"
              />
            )}
          </div>
        ))}
      </div>

      <button type="submit" disabled={loading} className="pbi-button w-full flex items-center justify-center gap-2 mt-6">
        {loading ? (
          <>
            <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            Analyzing...
          </>
        ) : (
          <>
            <FiSend className="text-lg" />
            Predict Severity
          </>
        )}
      </button>
    </form>
  );
};

export default PredictionForm;
