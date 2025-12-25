import React, { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  const [streaming, setStreaming] = useState(true);
  const [isDrawing, setIsDrawing] = useState(false);
  const [isDrawingLine, setIsDrawingLine] = useState(false);
  const [currentPoints, setCurrentPoints] = useState([]);
  const [linePoints, setLinePoints] = useState([]);
  const [zones, setZones] = useState([]);
  const [lines, setLines] = useState([]);
  const [zoneName, setZoneName] = useState('');
  const [lineName, setLineName] = useState('');
  const [showZones, setShowZones] = useState(true);
  
  const canvasRef = useRef(null);
  const imgRef = useRef(null);
  
  // URL to the FastAPI backend endpoint
  const API_URL = "http://localhost:8000";
  const STREAM_URL = `${API_URL}/video_feed`;

  // Load zones and lines on mount
  useEffect(() => {
    loadZones();
    loadLines();
    // Refresh line counts every 2 seconds
    const interval = setInterval(loadLines, 2000);
    return () => clearInterval(interval);
  }, []);

  const loadZones = async () => {
    try {
      const response = await fetch(`${API_URL}/api/zones`);
      const data = await response.json();
      setZones(data.zones || []);
    } catch (error) {
      console.error('Error loading zones:', error);
    }
  };

  const saveZone = async () => {
    if (currentPoints.length < 3) {
      alert('Please draw at least 3 points for a polygon zone');
      return;
    }

    const name = zoneName.trim() || `Zone ${zones.length + 1}`;
    const zone = {
      id: `zone_${Date.now()}`,
      name: name,
      points: currentPoints.map(p => ({ x: p.x, y: p.y })),
      color: '#00FF00',
      enabled: true
    };

    try {
      const response = await fetch(`${API_URL}/api/zones`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(zone)
      });
      
      if (response.ok) {
        await loadZones();
        setCurrentPoints([]);
        setIsDrawing(false);
        setZoneName('');
        alert('Zone saved successfully!');
      } else {
        const error = await response.json();
        alert(`Error: ${error.error || 'Failed to save zone'}`);
      }
    } catch (error) {
      console.error('Error saving zone:', error);
      alert('Failed to save zone');
    }
  };

  const deleteZone = async (zoneId) => {
    if (!confirm('Are you sure you want to delete this zone?')) return;

    try {
      await fetch(`${API_URL}/api/zones/${zoneId}`, { method: 'DELETE' });
      await loadZones();
    } catch (error) {
      console.error('Error deleting zone:', error);
    }
  };

  const deleteAllZones = async () => {
    if (!confirm('Are you sure you want to delete ALL zones?')) return;

    try {
      await fetch(`${API_URL}/api/zones`, { method: 'DELETE' });
      await loadZones();
    } catch (error) {
      console.error('Error deleting all zones:', error);
    }
  };

  // Counting Lines functions
  const loadLines = async () => {
    try {
      const response = await fetch(`${API_URL}/api/lines`);
      const data = await response.json();
      setLines(data.lines || []);
    } catch (error) {
      console.error('Error loading lines:', error);
    }
  };

  const saveLineWithPoints = async (points) => {
    if (points.length < 2) {
      alert('Please click 2 points to define the counting line');
      return;
    }

    const name = lineName.trim() || `Line ${lines.length + 1}`;
    const line = {
      id: `line_${Date.now()}`,
      name: name,
      start: { x: points[0].x, y: points[0].y },
      end: { x: points[1].x, y: points[1].y },
      color: '#FF0000',
      enabled: true
    };

    try {
      const response = await fetch(`${API_URL}/api/lines`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(line)
      });
      
      if (response.ok) {
        await loadLines();
        setLinePoints([]);
        setIsDrawingLine(false);
        setLineName('');
        alert('Counting line saved successfully!');
      } else {
        const error = await response.json();
        alert(`Error: ${error.error || 'Failed to save line'}`);
      }
    } catch (error) {
      console.error('Error saving line:', error);
      alert('Failed to save line');
    }
  };

  const deleteLine = async (lineId) => {
    if (!confirm('Are you sure you want to delete this line?')) return;

    try {
      await fetch(`${API_URL}/api/lines/${lineId}`, { method: 'DELETE' });
      await loadLines();
    } catch (error) {
      console.error('Error deleting line:', error);
    }
  };

  const resetLineCounts = async (lineId) => {
    try {
      await fetch(`${API_URL}/api/lines/${lineId}/reset`, { method: 'POST' });
      await loadLines();
    } catch (error) {
      console.error('Error resetting line counts:', error);
    }
  };

  const resetAllCounts = async () => {
    if (!confirm('Are you sure you want to reset ALL counts?')) return;

    try {
      await fetch(`${API_URL}/api/lines/reset-all`, { method: 'POST' });
      await loadLines();
    } catch (error) {
      console.error('Error resetting all counts:', error);
    }
  };

  const handleCanvasClick = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (isDrawing) {
      setCurrentPoints([...currentPoints, { x, y }]);
    } else if (isDrawingLine) {
      const newPoints = [...linePoints, { x, y }];
      setLinePoints(newPoints);
      // Auto-save after 2 points - pass the new points directly
      if (newPoints.length === 2) {
        saveLineWithPoints(newPoints);
      }
    }
  };

  const startDrawing = () => {
    setIsDrawing(true);
    setIsDrawingLine(false);
    setCurrentPoints([]);
  };

  const startDrawingLine = () => {
    setIsDrawingLine(true);
    setIsDrawing(false);
    setLinePoints([]);
  };

  const cancelDrawing = () => {
    setIsDrawing(false);
    setCurrentPoints([]);
  };

  const cancelDrawingLine = () => {
    setIsDrawingLine(false);
    setLinePoints([]);
  };

  const completePolygon = () => {
    if (currentPoints.length < 3) {
      alert('Please draw at least 3 points');
      return;
    }
    saveZone();
  };

  // Draw canvas overlay
  useEffect(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    
    if (!canvas || !img) return;

    const ctx = canvas.getContext('2d');
    canvas.width = img.width || 800;
    canvas.height = img.height || 600;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw current polygon being drawn
    if (currentPoints.length > 0) {
      ctx.strokeStyle = '#FFFF00';
      ctx.lineWidth = 3;
      ctx.fillStyle = 'rgba(255, 255, 0, 0.2)';

      ctx.beginPath();
      ctx.moveTo(currentPoints[0].x, currentPoints[0].y);
      for (let i = 1; i < currentPoints.length; i++) {
        ctx.lineTo(currentPoints[i].x, currentPoints[i].y);
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      // Draw points
      currentPoints.forEach((point, i) => {
        ctx.fillStyle = '#FFFF00';
        ctx.beginPath();
        ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.fillStyle = '#000';
        ctx.fillText(i + 1, point.x + 8, point.y - 8);
      });
    }

    // Draw counting line being drawn
    if (linePoints.length > 0) {
      ctx.strokeStyle = '#FF0000';
      ctx.lineWidth = 4;

      // Draw line if we have 2 points, or just the first point
      if (linePoints.length === 1) {
        ctx.fillStyle = '#FF0000';
        ctx.beginPath();
        ctx.arc(linePoints[0].x, linePoints[0].y, 6, 0, 2 * Math.PI);
        ctx.fill();
        ctx.fillStyle = '#FFF';
        ctx.fillText('1', linePoints[0].x + 10, linePoints[0].y - 10);
      } else if (linePoints.length === 2) {
        ctx.beginPath();
        ctx.moveTo(linePoints[0].x, linePoints[0].y);
        ctx.lineTo(linePoints[1].x, linePoints[1].y);
        ctx.stroke();
        
        // Draw endpoints
        linePoints.forEach((point, i) => {
          ctx.fillStyle = '#FF0000';
          ctx.beginPath();
          ctx.arc(point.x, point.y, 6, 0, 2 * Math.PI);
          ctx.fill();
          ctx.fillStyle = '#FFF';
          ctx.fillText(i + 1, point.x + 10, point.y - 10);
        });
      }
    }
  }, [currentPoints, linePoints]);

  return (
    <div className="container">
      <header>
        <h1>YOLO Real-time Detection with Zone Management</h1>
        <p>Source: {streaming ? "Live" : "Stopped"}</p>
      </header>

      <main className="video-container">
        <div style={{ position: 'relative', display: 'inline-block' }}>
          {streaming ? (
            <img 
              ref={imgRef}
              src={STREAM_URL} 
              alt="Real-time object detection stream"
              className="stream-image"
              onError={(e) => {
                console.error("Stream error", e);
              }}
            />
          ) : (
            <div className="placeholder">
              <p>Stream Paused</p>
            </div>
          )}
          
          {streaming && (
            <canvas
              ref={canvasRef}
              onClick={handleCanvasClick}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                cursor: (isDrawing || isDrawingLine) ? 'crosshair' : 'default',
                pointerEvents: (isDrawing || isDrawingLine) ? 'auto' : 'none'
              }}
            />
          )}
        </div>
      </main>

      <div className="controls">
        <div className="control-section">
          <h3>Stream Control</h3>
          <button onClick={() => setStreaming(!streaming)}>
            {streaming ? "Stop Stream" : "Start Stream"}
          </button>
        </div>

        <div className="control-section">
          <h3>Zone Drawing</h3>
          {!isDrawing ? (
            <button onClick={startDrawing} className="btn-primary">
              📐 Start Drawing Zone
            </button>
          ) : (
            <>
              <input
                type="text"
                placeholder="Zone name (optional)"
                value={zoneName}
                onChange={(e) => setZoneName(e.target.value)}
                style={{ marginBottom: '10px', padding: '5px' }}
              />
              <div>
                <button onClick={completePolygon} className="btn-success">
                  ✓ Complete Polygon ({currentPoints.length} points)
                </button>
                <button onClick={cancelDrawing} className="btn-danger">
                  ✗ Cancel
                </button>
              </div>
              <p style={{ fontSize: '12px', marginTop: '5px' }}>
                Click on the video to add points. Minimum 3 points required.
              </p>
            </>
          )}
        </div>

        <div className="control-section">
          <h3>Zones ({zones.length})</h3>
          {zones.length > 0 && (
            <button onClick={deleteAllZones} className="btn-danger" style={{ marginBottom: '10px' }}>
              🗑️ Delete All Zones
            </button>
          )}
          <div className="zones-list">
            {zones.map((zone) => (
              <div key={zone.id} className="zone-item">
                <span>{zone.name} ({zone.points.length} points)</span>
                <button onClick={() => deleteZone(zone.id)} className="btn-small btn-danger">
                  Delete
                </button>
              </div>
            ))}
            {zones.length === 0 && <p style={{ fontSize: '12px' }}>No zones defined. Draw a zone to start filtering detections.</p>}
          </div>
        </div>

        <div className="control-section">
          <h3>Counting Lines</h3>
          {!isDrawingLine ? (
            <button onClick={startDrawingLine} className="btn-primary">
              📏 Draw Counting Line
            </button>
          ) : (
            <>
              <input
                type="text"
                placeholder="Line name (optional)"
                value={lineName}
                onChange={(e) => setLineName(e.target.value)}
                style={{ marginBottom: '10px', padding: '5px' }}
              />
              <div>
                <button onClick={cancelDrawingLine} className="btn-danger">
                  ✗ Cancel
                </button>
              </div>
              <p style={{ fontSize: '12px', marginTop: '5px' }}>
                Click 2 points on the video to define the counting line.
                Point 1: Start, Point 2: End.
              </p>
            </>
          )}
        </div>

        <div className="control-section" style={{ gridColumn: 'span 2' }}>
          <h3>Traffic Density Monitor ({lines.length} lines)</h3>
          {lines.length > 0 && (
            <button onClick={resetAllCounts} className="btn-danger" style={{ marginBottom: '10px' }}>
              🔄 Reset All Data
            </button>
          )}
          <div className="density-container">
            {lines.map((line) => {
              const stats = line.stats || {};
              const rates = stats.rates || {};
              const directional = stats.directional || {};
              const congestion = stats.congestion || {};
              const trend = stats.trend || {};
              
              return (
                <div key={line.id} className="density-card">
                  {/* Header */}
                  <div className="density-header">
                    <div>
                      <strong>{line.name}</strong>
                      <span style={{ marginLeft: '10px', fontSize: '24px' }}>
                        {congestion.emoji || '🟢'}
                      </span>
                      <span style={{ 
                        marginLeft: '5px', 
                        fontSize: '14px', 
                        color: congestion.color || '#00FF00',
                        fontWeight: 'bold' 
                      }}>
                        {congestion.level || 'Low Traffic'}
                      </span>
                    </div>
                    <div style={{ fontSize: '18px' }}>
                      {trend.emoji || '→'} {trend.direction || 'Stable'}
                      {trend.change_pct !== undefined && trend.change_pct !== 0 && (
                        <span style={{ fontSize: '12px', marginLeft: '5px' }}>
                          ({trend.change_pct > 0 ? '+' : ''}{trend.change_pct}%)
                        </span>
                      )}
                    </div>
                  </div>
                  
                  {/* Density Rates */}
                  <div className="density-metrics">
                    <div className="metric">
                      <div className="metric-label">Real-time (1 min)</div>
                      <div className="metric-value">{rates.current || 0} veh/min</div>
                    </div>
                    <div className="metric">
                      <div className="metric-label">Short-term (5 min)</div>
                      <div className="metric-value">{rates.short || 0} veh/min</div>
                    </div>
                    <div className="metric">
                      <div className="metric-label">Medium-term (15 min)</div>
                      <div className="metric-value">{rates.medium || 0} veh/min</div>
                    </div>
                    <div className="metric">
                      <div className="metric-label">Long-term (1 hour)</div>
                      <div className="metric-value">{rates.long || 0} veh/min</div>
                    </div>
                  </div>
                  
                  {/* Directional Split */}
                  <div className="directional-split">
                    <strong style={{ fontSize: '12px' }}>Directional (5-min avg):</strong>
                    <div style={{ display: 'flex', gap: '10px', marginTop: '5px', fontSize: '12px' }}>
                      <span>↑ {directional.up || 0}</span>
                      <span>↓ {directional.down || 0}</span>
                      <span>← {directional.left || 0}</span>
                      <span>→ {directional.right || 0}</span>
                      <span style={{ marginLeft: 'auto' }}>veh/min</span>
                    </div>
                  </div>
                  
                  {/* History Chart (Simple Bar) */}
                  {stats.history && stats.history.length > 0 && (
                    <div className="history-chart">
                      <div style={{ fontSize: '10px', marginBottom: '3px' }}>Last {stats.history.length} minutes:</div>
                      <div style={{ display: 'flex', gap: '1px', height: '30px', alignItems: 'flex-end' }}>
                        {stats.history.slice(-30).map((count, idx) => {
                          const maxInHistory = Math.max(...stats.history, 1);
                          const height = (count / maxInHistory) * 100;
                          return (
                            <div
                              key={idx}
                              style={{
                                flex: 1,
                                background: count > line.threshold_high ? '#FF0000' : 
                                           count > line.threshold_normal ? '#FFA500' : 
                                           count > line.threshold_low ? '#FFFF00' : '#00FF00',
                                height: `${height}%`,
                                minHeight: count > 0 ? '2px' : '0px'
                              }}
                              title={`${count} vehicles`}
                            />
                          );
                        })}
                      </div>
                    </div>
                  )}
                  
                  {/* Actions */}
                  <div style={{ marginTop: '10px', display: 'flex', gap: '5px' }}>
                    <button onClick={() => resetLineCounts(line.id)} className="btn-small" style={{background: '#2196F3'}}>
                      🔄 Reset
                    </button>
                    <button onClick={() => deleteLine(line.id)} className="btn-small btn-danger">
                      🗑️ Delete
                    </button>
                  </div>
                </div>
              );
            })}
            {lines.length === 0 && (
              <p style={{ fontSize: '12px', gridColumn: 'span 2' }}>
                No counting lines defined. Draw a line across the road to start monitoring traffic density.
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;

