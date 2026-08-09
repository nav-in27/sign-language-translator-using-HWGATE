import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer, Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis } from 'recharts';
import jsPDF from 'jspdf';
import 'jspdf-autotable';
import Papa from 'papaparse';
import '../index.css';

function Benchmark() {
  const [status, setStatus] = useState({ message: "Idle", progress: 0 });
  const [resultsData, setResultsData] = useState(null);
  const [isRunning, setIsRunning] = useState(false);

  useEffect(() => {
    // Check initial results if they exist
    fetchResults();
  }, []);

  const fetchResults = async () => {
    try {
      const res = await fetch("http://localhost:8000/benchmark/results");
      const data = await res.json();
      if (!data.error) {
        setResultsData(data);
      }
    } catch (e) {
      console.log("No previous results found");
    }
  };

  const runBenchmark = async () => {
    setIsRunning(true);
    setStatus({ message: "Starting Benchmark...", progress: 0 });
    
    try {
      await fetch("http://localhost:8000/benchmark/run", { method: "POST" });
      
      const interval = setInterval(async () => {
        try {
          const res = await fetch("http://localhost:8000/benchmark/status");
          const data = await res.json();
          setStatus(data);
          
          if (data.progress >= 100) {
            clearInterval(interval);
            setIsRunning(false);
            fetchResults();
          }
        } catch (e) {
          console.error(e);
        }
      }, 2000);
    } catch (e) {
      setIsRunning(false);
      setStatus({ message: "Failed to start", progress: 0 });
    }
  };

  const exportPDF = () => {
    if (!resultsData) return;
    const doc = new jsPDF();
    doc.text("Live AI Benchmark Report", 14, 15);
    doc.text(`Evaluation Time: ${resultsData.evaluation_time}`, 14, 22);
    
    const tableColumn = ["Model", "Accuracy", "F1 Score", "Latency (ms)", "FPS", "Model Size (MB)"];
    const tableRows = [];

    resultsData.results.forEach(r => {
      tableRows.push([
        r.model,
        (r.accuracy * 100).toFixed(2) + "%",
        (r.f1 * 100).toFixed(2) + "%",
        r.latency_ms.toFixed(2),
        r.fps.toFixed(1),
        r.model_size_mb.toFixed(2)
      ]);
    });

    doc.autoTable({
      head: [tableColumn],
      body: tableRows,
      startY: 30,
    });
    
    doc.save("benchmark_report.pdf");
  };

  const exportCSV = () => {
    if (!resultsData) return;
    const csv = Papa.unparse(resultsData.results);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', 'benchmark_report.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Radar chart normalization
  const radarData = resultsData?.results.map(r => ({
    subject: r.model,
    Accuracy: r.accuracy * 100,
    F1: r.f1 * 100,
    "Efficiency (FPS/10)": Math.min(r.fps / 10, 100),
    "Compactness": Math.max(100 - r.model_size_mb * 5, 0)
  }));

  return (
    <div className="benchmark-container">
      <header className="benchmark-header">
        <h1>Model Evaluation Benchmark</h1>
        <p>Measured Live Results</p>
      </header>

      <div className="controls">
        <button className="run-btn" onClick={runBenchmark} disabled={isRunning}>
          {isRunning ? "Benchmarking..." : "▶ Run Live Benchmark"}
        </button>
        {resultsData && (
          <>
            <button className="export-btn" onClick={exportPDF}>📥 Export PDF</button>
            <button className="export-btn" onClick={exportCSV}>📥 Export CSV</button>
          </>
        )}
      </div>

      {isRunning && (
        <div className="progress-bar-container">
          <div className="progress-bar" style={{ width: `${status.progress}%` }}></div>
          <p>{status.progress}% - {status.message}</p>
        </div>
      )}

      {resultsData && (
        <div className="results-content">
          <div className="dataset-info">
            <h3>Dataset Info</h3>
            <p>Total Samples: {resultsData.dataset_info.total_samples}</p>
            <p>Train/Test Split: {resultsData.dataset_info.train_samples} / {resultsData.dataset_info.test_samples}</p>
            <p>Classes: {resultsData.dataset_info.num_classes}</p>
            <p>Evaluation Time: {resultsData.evaluation_time}</p>
          </div>

          <div className="metrics-cards">
            {resultsData.results.map((r, i) => (
              <div key={i} className={`metric-card ${r.is_best ? 'best-model' : ''}`}>
                <h3>{r.model} {r.is_best && '🏆 (Best)'}</h3>
                <p>Accuracy: {(r.accuracy * 100).toFixed(1)}%</p>
                <p>F1 Score: {(r.f1 * 100).toFixed(1)}%</p>
                <p>Latency: {r.latency_ms.toFixed(2)} ms</p>
                <p>FPS: {r.fps.toFixed(1)}</p>
                <p>Size: {r.model_size_mb.toFixed(2)} MB</p>
              </div>
            ))}
          </div>

          <div className="charts">
            <div className="chart-box">
              <h3>Accuracy & F1 Score</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={resultsData.results}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2e3a59" />
                  <XAxis dataKey="model" stroke="#8e8e93" />
                  <YAxis domain={[0, 1]} stroke="#8e8e93" />
                  <RechartsTooltip contentStyle={{ backgroundColor: '#1a2235', border: '1px solid rgba(255,255,255,0.1)' }} />
                  <Legend />
                  <Bar dataKey="accuracy" fill="#ccff00" name="Accuracy" />
                  <Bar dataKey="f1" fill="#c98ff6" name="F1 Score" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-box">
              <h3>Model Capabilities (Radar)</h3>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart outerRadius={90} data={radarData}>
                  <PolarGrid stroke="#2e3a59" />
                  <PolarAngleAxis dataKey="subject" stroke="#8e8e93" />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="#2e3a59" />
                  <Radar name="Accuracy" dataKey="Accuracy" stroke="#ccff00" fill="#ccff00" fillOpacity={0.5} />
                  <Radar name="F1" dataKey="F1" stroke="#c98ff6" fill="#c98ff6" fillOpacity={0.5} />
                  <Legend />
                  <RechartsTooltip contentStyle={{ backgroundColor: '#1a2235', border: '1px solid rgba(255,255,255,0.1)' }} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Benchmark;
