import React, { useState } from 'react';
import './App.css';

function App() {
  const [streaming, setStreaming] = useState(true);
  
  // URL to the FastAPI backend endpoint
  // Assuming backend runs on port 8000
  const STREAM_URL = "http://localhost:8000/video_feed";

  return (
    <div className="container">
      <header>
        <h1>YOLO Real-time Detection</h1>
        <p>Source: {streaming ? "Live" : "Stopped"}</p>
      </header>

      <main className="video-container">
        {streaming ? (
          <img 
            src={STREAM_URL} 
            alt="Real-time object detection stream"
            className="stream-image"
            onError={(e) => {
              console.error("Stream error", e);
              // Optional: Add retry logic or placeholder here
            }}
          />
        ) : (
          <div className="placeholder">
            <p>Stream Paused</p>
          </div>
        )}
      </main>

      <div className="controls">
        <button onClick={() => setStreaming(!streaming)}>
          {streaming ? "Stop Stream" : "Start Stream"}
        </button>
      </div>
    </div>
  );
}

export default App;

