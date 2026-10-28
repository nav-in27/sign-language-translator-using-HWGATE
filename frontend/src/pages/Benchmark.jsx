import React, { useEffect, useMemo, useState } from 'react';
import { API_BASE_URL } from '../config';
import '../index.css';

const percent = (val) => val == null ? '—' : `${(val * 100).toFixed(2)}%`;

function MetricCard({ title, value, subtitle, best, highlight }) {
  return (
    <div className={`metric-card ${best ? 'best-model' : ''} ${highlight ? 'highlight-metric' : ''}`}>
      <span className="metric-label">{title}</span>
      <strong className="metric-value">{value}</strong>
      {subtitle && <span className="metric-sub">{subtitle}</span>}
    </div>
  );
}

function Benchmark() {
  const [comparisonData, setComparisonData] = useState(null);
  const [confusionData, setConfusionData] = useState(null);
  const [selectedModel, setSelectedModel] = useState('AC-HWGATE (Production SOTA)');
  const [matrixMode, setMatrixMode] = useState('norm'); // 'norm' | 'raw'
  const [matrixView, setMatrixView] = useState('top25'); // 'top25' | 'full'
  const [classFilter, setClassFilter] = useState('');
  const [tableFilter, setTableFilter] = useState('all'); // 'all' | 'perfect' | 'weak'
  const [tableSearch, setTableSearch] = useState('');
  const [hoveredCell, setHoveredCell] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const loadData = async () => {
    setLoading(true);
    try {
      const [compRes, cmRes] = await Promise.all([
        fetch(`${API_BASE_URL}/benchmark/comparison`, { cache: 'no-store' }),
        fetch(`${API_BASE_URL}/benchmark/confusion_matrix`, { cache: 'no-store' })
      ]);

      if (!compRes.ok) throw new Error('Failed loading benchmark comparison');
      const comp = await compRes.json();
      setComparisonData(comp);

      if (cmRes.ok) {
        const cm = await cmRes.json();
        setConfusionData(cm);
      }
      setError('');
    } catch (err) {
      setError(err.message || 'Error loading benchmark data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const modelsList = useMemo(() => {
    return comparisonData?.models || [];
  }, [comparisonData]);

  const activeModelSummary = useMemo(() => {
    return modelsList.find(m => m.model_name === selectedModel) || modelsList[0] || null;
  }, [modelsList, selectedModel]);

  const activeModelCM = useMemo(() => {
    if (!confusionData?.models) return null;
    return confusionData.models[selectedModel] || confusionData.models[modelsList[0]?.model_name] || null;
  }, [confusionData, selectedModel, modelsList]);

  const classLabels = useMemo(() => {
    return confusionData?.class_labels || [];
  }, [confusionData]);

  // Classes to display in confusion matrix
  const matrixClasses = useMemo(() => {
    if (!activeModelCM || !classLabels.length) return [];
    if (matrixView === 'top25') {
      // Pick top 25 classes with highest test support or most confusion pairs
      const topPairs = activeModelCM.top_confusion_pairs || [];
      const prioritySet = new Set();
      topPairs.slice(0, 15).forEach(p => {
        prioritySet.add(p.actual_idx);
        prioritySet.add(p.predicted_idx);
      });
      // Fill rest up to 25 from per_class
      const sortedPerClass = [...(activeModelCM.per_class || [])].sort((a, b) => b.support - a.support);
      for (const item of sortedPerClass) {
        if (prioritySet.size >= 25) break;
        prioritySet.add(item.class_idx);
      }
      return Array.from(prioritySet).sort((a, b) => a - b);
    }
    // Full view: all classes or filter by search query
    if (classFilter.trim()) {
      const q = classFilter.toLowerCase().trim();
      return classLabels
        .map((name, idx) => ({ name, idx }))
        .filter(item => item.name.toLowerCase().includes(q))
        .map(item => item.idx);
    }
    return classLabels.map((_, idx) => idx);
  }, [activeModelCM, classLabels, matrixView, classFilter]);

  // Per-class filtered list
  const filteredPerClass = useMemo(() => {
    if (!activeModelCM?.per_class) return [];
    let list = activeModelCM.per_class;
    if (tableFilter === 'perfect') {
      list = list.filter(item => item.f1 >= 0.99);
    } else if (tableFilter === 'weak') {
      list = list.filter(item => item.f1 < 0.50);
    }
    if (tableSearch.trim()) {
      const q = tableSearch.toLowerCase().trim();
      list = list.filter(item => item.class_name.toLowerCase().includes(q));
    }
    return list;
  }, [activeModelCM, tableFilter, tableSearch]);

  return (
    <div className="benchmark-dashboard-container">
      {/* Header */}
      <header className="benchmark-header">
        <div className="header-title-row">
          <div>
            <h1 className="benchmark-main-title">MODEL COMPARISON &amp; BENCHMARK DASHBOARD</h1>
            <p className="benchmark-subtitle">
              Authentic multi-model evaluation across all 136 ISL classes on clean held-out test split (722 samples)
            </p>
          </div>
          <button className="refresh-btn" onClick={loadData} disabled={loading}>
            {loading ? 'Evaluating...' : '↻ Refresh Data'}
          </button>
        </div>

        {/* Model Tabs */}
        {modelsList.length > 0 && (
          <div className="model-tabs-bar" role="tablist">
            {modelsList.map(m => (
              <button
                key={m.model_name}
                className={`model-tab ${selectedModel === m.model_name ? 'active' : ''}`}
                onClick={() => setSelectedModel(m.model_name)}
                role="tab"
                aria-selected={selectedModel === m.model_name}
              >
                <span className="tab-name">{m.model_name}</span>
                <span className="tab-acc">{(m.top1_accuracy * 100).toFixed(1)}%</span>
              </button>
            ))}
          </div>
        )}
      </header>

      {error && <div className="benchmark-alert error">{error}</div>}
      {loading && !comparisonData && <div className="benchmark-alert loading">Loading benchmark evaluation...</div>}

      {activeModelSummary && (
        <section className="summary-cards-section">
          <div className="metrics-grid">
            <MetricCard
              title="Top-1 Accuracy"
              value={percent(activeModelSummary.top1_accuracy)}
              subtitle={`${activeModelSummary.sample_count} test sequences`}
              best={selectedModel.includes('Production SOTA') || activeModelSummary.top1_accuracy >= 0.91}
              highlight
            />
            <MetricCard
              title="Top-5 Accuracy"
              value={percent(activeModelSummary.top5_accuracy)}
              subtitle="Rank-5 coverage"
            />
            <MetricCard
              title="Macro F1 Score"
              value={percent(activeModelSummary.macro_f1)}
              subtitle={`Macro Prec: ${percent(activeModelSummary.macro_precision)}`}
            />
            <MetricCard
              title="Weighted F1"
              value={percent(activeModelSummary.weighted_f1)}
              subtitle={`Macro Rec: ${percent(activeModelSummary.macro_recall)}`}
            />
            <MetricCard
              title="Inference Latency"
              value={`${activeModelSummary.mean_latency_ms} ms`}
              subtitle={`p50: ${activeModelSummary.p50_latency_ms}ms · p95: ${activeModelSummary.p95_latency_ms}ms`}
            />
            <MetricCard
              title="Throughput (FPS)"
              value={`${activeModelSummary.throughput_fps}`}
              subtitle="Sustained real-time FPS"
            />
            <MetricCard
              title="Parameters &amp; Size"
              value={`${(activeModelSummary.parameters / 1000).toFixed(0)}k params`}
              subtitle={`Artifact: ${activeModelSummary.artifact_size_mb} MB`}
            />
          </div>
        </section>
      )}

      {/* Side-by-Side Model Comparison Table */}
      <section className="section-card">
        <div className="section-header">
          <h2>Rigorous Model Comparison (Identical 136-Class Test Split)</h2>
          <span className="badge-pill">722 Samples · Clean Held-Out Split</span>
        </div>
        <div className="table-responsive">
          <table className="comparison-table">
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>Classes</th>
                <th>Top-1 Acc</th>
                <th>Top-5 Acc</th>
                <th>Macro F1</th>
                <th>Weighted F1</th>
                <th>Macro Prec</th>
                <th>Macro Rec</th>
                <th>Mean Latency</th>
                <th>p95 Latency</th>
                <th>FPS</th>
                <th>Parameters</th>
                <th>Artifact Size</th>
              </tr>
            </thead>
            <tbody>
              {modelsList.map(m => {
                const isSelected = m.model_name === selectedModel;
                const isSota = m.model_name.includes('Production SOTA');
                return (
                  <tr
                    key={m.model_name}
                    className={`${isSelected ? 'selected-row' : ''} ${isSota ? 'sota-row' : ''}`}
                    onClick={() => setSelectedModel(m.model_name)}
                    style={{ cursor: 'pointer' }}
                  >
                    <td>
                      <strong>{m.model_name}</strong>
                      {isSota && <span className="sota-badge">PRODUCTION SOTA</span>}
                    </td>
                    <td>{m.num_classes}</td>
                    <td><span className="metric-cell-bold">{percent(m.top1_accuracy)}</span></td>
                    <td>{percent(m.top5_accuracy)}</td>
                    <td><span className="metric-cell-bold">{percent(m.macro_f1)}</span></td>
                    <td>{percent(m.weighted_f1)}</td>
                    <td>{percent(m.macro_precision)}</td>
                    <td>{percent(m.macro_recall)}</td>
                    <td>{m.mean_latency_ms} ms</td>
                    <td>{m.p95_latency_ms} ms</td>
                    <td><span className="fps-cell">{m.throughput_fps}</span></td>
                    <td>{Number(m.parameters).toLocaleString()}</td>
                    <td>{m.artifact_size_mb} MB</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* Visual Analytics Charts */}
      <section className="section-card">
        <div className="section-header">
          <h2>Visual Analytics: Accuracy, F1 &amp; Efficiency Across Architectures</h2>
        </div>
        <div className="charts-grid">
          {/* Chart 1: Top-1 vs Macro F1 */}
          <div className="chart-box">
            <h3>Top-1 Accuracy vs Macro F1 Score</h3>
            <div className="model-bars-container">
              {modelsList.map(m => (
                <div className="model-bar-item" key={m.model_name}>
                  <div className="bar-header">
                    <span className="bar-label">{m.model_name}</span>
                    <span className="bar-val">{percent(m.top1_accuracy)} (F1: {percent(m.macro_f1)})</span>
                  </div>
                  <div className="stacked-tracks">
                    <div className="track-bar">
                      <div
                        className="track-fill acc"
                        style={{ width: `${Math.min(100, m.top1_accuracy * 100)}%` }}
                        title={`Accuracy: ${percent(m.top1_accuracy)}`}
                      />
                    </div>
                    <div className="track-bar f1">
                      <div
                        className="track-fill f1"
                        style={{ width: `${Math.min(100, m.macro_f1 * 100)}%` }}
                        title={`Macro F1: ${percent(m.macro_f1)}`}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Chart 2: Latency & Throughput */}
          <div className="chart-box">
            <h3>Inference Latency &amp; Sustained Throughput (FPS)</h3>
            <div className="model-bars-container">
              {modelsList.map(m => {
                const maxFps = Math.max(...modelsList.map(x => x.throughput_fps || 1));
                const fpsPct = Math.min(100, ((m.throughput_fps || 0) / maxFps) * 100);
                return (
                  <div className="model-bar-item" key={m.model_name}>
                    <div className="bar-header">
                      <span className="bar-label">{m.model_name}</span>
                      <span className="bar-val">{m.mean_latency_ms} ms · {m.throughput_fps} FPS</span>
                    </div>
                    <div className="track-bar fps">
                      <div
                        className="track-fill fps"
                        style={{ width: `${fpsPct}%` }}
                        title={`${m.throughput_fps} FPS`}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </section>

      {/* Interactive Confusion Matrix */}
      {activeModelCM && (
        <section className="section-card confusion-matrix-section">
          <div className="section-header">
            <div>
              <h2>Interactive 136 × 136 Confusion Matrix</h2>
              <p className="section-desc">
                Predictions for <strong>{selectedModel}</strong> ({classLabels.length} classes, {activeModelSummary?.sample_count} test samples)
              </p>
            </div>
            <div className="matrix-controls">
              <div className="btn-group">
                <button
                  className={`toggle-btn ${matrixMode === 'norm' ? 'active' : ''}`}
                  onClick={() => setMatrixMode('norm')}
                >
                  Row-Normalized %
                </button>
                <button
                  className={`toggle-btn ${matrixMode === 'raw' ? 'active' : ''}`}
                  onClick={() => setMatrixMode('raw')}
                >
                  Raw Counts
                </button>
              </div>

              <div className="btn-group">
                <button
                  className={`toggle-btn ${matrixView === 'top25' ? 'active' : ''}`}
                  onClick={() => setMatrixView('top25')}
                >
                  Top 25 Active
                </button>
                <button
                  className={`toggle-btn ${matrixView === 'full' ? 'active' : ''}`}
                  onClick={() => setMatrixView('full')}
                >
                  Full Matrix
                </button>
              </div>

              {matrixView === 'full' && (
                <input
                  type="text"
                  placeholder="Filter class..."
                  className="filter-input"
                  value={classFilter}
                  onChange={e => setClassFilter(e.target.value)}
                />
              )}
            </div>
          </div>

          {hoveredCell && (
            <div className="matrix-tooltip-banner">
              <strong>Actual:</strong> {classLabels[hoveredCell.actual]} &nbsp;→&nbsp;
              <strong>Predicted:</strong> {classLabels[hoveredCell.pred]} &nbsp;|&nbsp;
              <strong>Count:</strong> {hoveredCell.count} &nbsp;|&nbsp;
              <strong>Rate:</strong> {(hoveredCell.rate * 100).toFixed(1)}%
              {hoveredCell.actual === hoveredCell.pred && <span className="correct-badge">CORRECT MATCH</span>}
            </div>
          )}

          <div className="matrix-scroll-wrapper">
            <div className="matrix-grid-table">
              <div className="matrix-header-row">
                <div className="matrix-corner-cell">Actual \ Pred</div>
                {matrixClasses.map(pIdx => (
                  <div key={`col-${pIdx}`} className="matrix-header-col" title={classLabels[pIdx]}>
                    <span>{classLabels[pIdx]}</span>
                  </div>
                ))}
              </div>

              {matrixClasses.map(aIdx => (
                <div key={`row-${aIdx}`} className="matrix-row">
                  <div className="matrix-header-label" title={classLabels[aIdx]}>
                    {classLabels[aIdx]}
                  </div>
                  {matrixClasses.map(pIdx => {
                    const rawVal = activeModelCM.confusion_matrix_raw?.[aIdx]?.[pIdx] || 0;
                    const normVal = activeModelCM.confusion_matrix_norm?.[aIdx]?.[pIdx] || 0;
                    const isDiag = aIdx === pIdx;
                    const displayVal = matrixMode === 'norm'
                      ? (normVal > 0 ? `${(normVal * 100).toFixed(0)}%` : '')
                      : (rawVal > 0 ? rawVal : '');

                    // Color intensity
                    const intensity = matrixMode === 'norm' ? normVal : Math.min(1, rawVal / 8);
                    const bgColor = isDiag
                      ? `rgba(16, 185, 129, ${Math.max(0.08, intensity * 0.9)})`
                      : rawVal > 0
                        ? `rgba(239, 68, 68, ${Math.max(0.15, intensity * 0.85)})`
                        : 'transparent';

                    return (
                      <div
                        key={`cell-${aIdx}-${pIdx}`}
                        className={`matrix-cell ${isDiag ? 'diagonal' : ''} ${rawVal > 0 && !isDiag ? 'error-cell' : ''}`}
                        style={{ backgroundColor: bgColor }}
                        onMouseEnter={() => setHoveredCell({ actual: aIdx, pred: pIdx, count: rawVal, rate: normVal })}
                        onMouseLeave={() => setHoveredCell(null)}
                      >
                        {displayVal}
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          </div>
        </section>
      )}

      {/* Top Confusion Pairs & Analysis */}
      {activeModelCM?.top_confusion_pairs?.length > 0 && (
        <section className="section-card">
          <div className="section-header">
            <h2>Most Frequent Confusion Pairs ({selectedModel})</h2>
            <span className="badge-pill">Fine-grained Gesture Differentiation</span>
          </div>
          <div className="confusion-pairs-grid">
            {activeModelCM.top_confusion_pairs.slice(0, 12).map((pair, idx) => (
              <div className="confusion-pair-card" key={idx}>
                <div className="pair-words">
                  <span className="actual-word">{pair.actual_class}</span>
                  <span className="arrow">→</span>
                  <span className="pred-word">{pair.predicted_class}</span>
                </div>
                <div className="pair-stats">
                  <span className="pair-count">{pair.count} errors</span>
                  <span className="pair-rate">{(pair.rate * 100).toFixed(1)}% misclassified</span>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Per-Class 136 Breakdown */}
      {activeModelCM?.per_class && (
        <section className="section-card">
          <div className="section-header">
            <div>
              <h2>Per-Class Evaluation (All 136 ISL Classes)</h2>
              <p className="section-desc">Class support, Precision, Recall, and F1 on clean held-out test split</p>
            </div>
            <div className="per-class-filters">
              <div className="btn-group">
                <button
                  className={`toggle-btn ${tableFilter === 'all' ? 'active' : ''}`}
                  onClick={() => setTableFilter('all')}
                >
                  All ({activeModelCM.per_class.length})
                </button>
                <button
                  className={`toggle-btn ${tableFilter === 'perfect' ? 'active' : ''}`}
                  onClick={() => setTableFilter('perfect')}
                >
                  Perfect F1 ({activeModelCM.per_class.filter(c => c.f1 >= 0.99).length})
                </button>
                <button
                  className={`toggle-btn ${tableFilter === 'weak' ? 'active' : ''}`}
                  onClick={() => setTableFilter('weak')}
                >
                  Weak F1 &lt; 0.5 ({activeModelCM.per_class.filter(c => c.f1 < 0.50).length})
                </button>
              </div>
              <input
                type="text"
                placeholder="Search class name..."
                className="filter-input"
                value={tableSearch}
                onChange={e => setTableSearch(e.target.value)}
              />
            </div>
          </div>

          <div className="table-responsive">
            <table className="per-class-table">
              <thead>
                <tr>
                  <th>Index</th>
                  <th>Class Name</th>
                  <th>Test Samples</th>
                  <th>TP</th>
                  <th>FP</th>
                  <th>FN</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1 Score</th>
                </tr>
              </thead>
              <tbody>
                {filteredPerClass.map(item => {
                  const isPerfect = item.f1 >= 0.99;
                  const isWeak = item.f1 < 0.50;
                  return (
                    <tr key={item.class_idx} className={isPerfect ? 'perfect-row' : isWeak ? 'weak-row' : ''}>
                      <td>{item.class_idx}</td>
                      <td><strong>{item.class_name}</strong></td>
                      <td>{item.support}</td>
                      <td>{item.tp}</td>
                      <td>{item.fp}</td>
                      <td>{item.fn}</td>
                      <td>{percent(item.precision)}</td>
                      <td>{percent(item.recall)}</td>
                      <td>
                        <span className={`f1-pill ${isPerfect ? 'perfect' : isWeak ? 'weak' : 'good'}`}>
                          {percent(item.f1)}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}

export default Benchmark;
