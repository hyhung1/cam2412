import os
# Fix OpenMP conflict issue - must be set before importing other libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO
import threading
import time
import atexit
from contextlib import asynccontextmanager
import numpy as np
import json
from typing import List, Dict, Optional
from pydantic import BaseModel
import torch
from datetime import datetime, timedelta
from collections import deque

# Pydantic models for API
class Point(BaseModel):
    x: float
    y: float

class Zone(BaseModel):
    id: str
    name: str
    points: List[Point]
    color: Optional[str] = "#00FF00"  # Default green
    enabled: Optional[bool] = True

class CountingLine(BaseModel):
    id: str
    name: str
    start: Point
    end: Point
    color: Optional[str] = "#FF0000"  # Default red
    enabled: Optional[bool] = True
    # Congestion thresholds (vehicles per minute)
    threshold_low: Optional[float] = 5.0
    threshold_normal: Optional[float] = 15.0
    threshold_high: Optional[float] = 25.0

# Global event to control the streaming loop
shutdown_event = threading.Event()

# Global list to track all video capture instances
active_captures = []
capture_lock = threading.Lock()

# Global zone storage
detection_zones: List[Zone] = []
zones_lock = threading.Lock()
ZONES_FILE = "detection_zones.json"

# Global counting line storage
counting_lines: List[CountingLine] = []
lines_lock = threading.Lock()
LINES_FILE = "counting_lines.json"

# Vehicle tracking for line crossing detection
vehicle_positions = {}  # {track_id: [(x, y, frame_num), ...]}
positions_lock = threading.Lock()

# NEW: Time-windowed crossing data with timestamps
# {line_id: {"up": deque[(timestamp, track_id), ...], "down": deque[...], ...}}
crossing_history = {}
history_lock = threading.Lock()
MAX_HISTORY_SECONDS = 3600  # Keep 1 hour of history

# NEW: Minute-bucketed counts for visualization
# {line_id: {"buckets": deque[count, count, ...], "current_minute": datetime, "current_count": int}}
minute_buckets = {}
buckets_lock = threading.Lock()
MAX_BUCKETS = 60  # Keep 60 minutes of history

# Stats caching to avoid expensive calculations on every API call
stats_cache = {}  # {line_id: {"stats": {...}, "timestamp": datetime}}
stats_cache_lock = threading.Lock()
STATS_CACHE_SECONDS = 5  # Refresh stats every 5 seconds

def load_zones():
    """Load zones from file"""
    global detection_zones
    try:
        if os.path.exists(ZONES_FILE):
            with open(ZONES_FILE, 'r') as f:
                data = json.load(f)
                with zones_lock:
                    detection_zones = [Zone(**zone) for zone in data]
                print(f"Loaded {len(detection_zones)} detection zones")
    except Exception as e:
        print(f"Error loading zones: {e}")
        detection_zones = []

def save_zones():
    """Save zones to file"""
    try:
        with zones_lock:
            data = [zone.model_dump() for zone in detection_zones]
        with open(ZONES_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(detection_zones)} detection zones")
    except Exception as e:
        print(f"Error saving zones: {e}")

def initialize_line_data_structures(line_id):
    """Initialize all data structures for a counting line"""
    with history_lock:
        if line_id not in crossing_history:
            crossing_history[line_id] = {
                "up": deque(),
                "down": deque(),
                "left": deque(),
                "right": deque()
            }
    
    with buckets_lock:
        if line_id not in minute_buckets:
            minute_buckets[line_id] = {
                "buckets": deque(maxlen=MAX_BUCKETS),  # Last 60 minutes
                "current_minute": datetime.now().replace(second=0, microsecond=0),
                "current_count": 0
            }

def load_lines():
    """Load counting lines from file"""
    global counting_lines
    try:
        if os.path.exists(LINES_FILE):
            with open(LINES_FILE, 'r') as f:
                data = json.load(f)
                with lines_lock:
                    counting_lines = [CountingLine(**line) for line in data]
                    # Initialize data structures for each line
                    for line in counting_lines:
                        initialize_line_data_structures(line.id)
                print(f"Loaded {len(counting_lines)} counting lines")
    except Exception as e:
        print(f"Error loading lines: {e}")
        counting_lines = []

def save_lines():
    """Save counting lines to file"""
    try:
        with lines_lock:
            data = [line.model_dump() for line in counting_lines]
        with open(LINES_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(counting_lines)} counting lines")
    except Exception as e:
        print(f"Error saving lines: {e}")

def clean_old_crossing_data():
    """Remove crossing data older than MAX_HISTORY_SECONDS"""
    now = datetime.now()
    cutoff_time = now - timedelta(seconds=MAX_HISTORY_SECONDS)
    
    with history_lock:
        for line_id in crossing_history:
            for direction in ["up", "down", "left", "right"]:
                # Remove old entries
                while crossing_history[line_id][direction] and crossing_history[line_id][direction][0][0] < cutoff_time:
                    crossing_history[line_id][direction].popleft()

def update_minute_buckets_internal(line_id):
    """Update minute buckets - CALLER MUST HOLD buckets_lock"""
    if line_id not in minute_buckets:
        return
    
    now = datetime.now().replace(second=0, microsecond=0)
    bucket_data = minute_buckets[line_id]
    
    # Check if we've moved to a new minute
    if now > bucket_data["current_minute"]:
        # Save current count to buckets
        bucket_data["buckets"].append(bucket_data["current_count"])
        # Start new minute
        bucket_data["current_minute"] = now
        bucket_data["current_count"] = 0

def get_vehicles_per_minute(line_id, window_seconds):
    """Calculate vehicles per minute for a given time window"""
    now = datetime.now()
    cutoff_time = now - timedelta(seconds=window_seconds)
    
    total_count = 0
    with history_lock:
        if line_id in crossing_history:
            for direction in ["up", "down", "left", "right"]:
                for timestamp, track_id in crossing_history[line_id][direction]:
                    if timestamp >= cutoff_time:
                        total_count += 1
    
    # Convert to vehicles per minute
    window_minutes = window_seconds / 60.0
    return total_count / window_minutes if window_minutes > 0 else 0.0

def get_directional_density(line_id, window_seconds):
    """Get vehicles per minute for each direction"""
    now = datetime.now()
    cutoff_time = now - timedelta(seconds=window_seconds)
    window_minutes = window_seconds / 60.0
    
    result = {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0}
    
    with history_lock:
        if line_id in crossing_history:
            for direction in ["up", "down", "left", "right"]:
                count = sum(1 for timestamp, track_id in crossing_history[line_id][direction] if timestamp >= cutoff_time)
                result[direction] = count / window_minutes if window_minutes > 0 else 0.0
    
    return result

def get_congestion_level(rate, thresholds):
    """Determine congestion level based on rate and thresholds
    Returns: ("level_name", "color", "emoji")
    """
    if rate < thresholds["low"]:
        return ("Low Traffic", "#00FF00", "🟢")
    elif rate < thresholds["normal"]:
        return ("Normal", "#FFFF00", "🟡")
    elif rate < thresholds["high"]:
        return ("Heavy Traffic", "#FFA500", "🟠")
    else:
        return ("TRAFFIC JAM", "#FF0000", "🔴")

def get_trend(line_id, recent_window, previous_window):
    """Detect trend by comparing two time windows
    Returns: ("trend", "emoji", percentage_change)
    """
    recent_rate = get_vehicles_per_minute(line_id, recent_window)
    
    # Get previous window (same duration, but shifted back in time)
    now = datetime.now()
    previous_cutoff = now - timedelta(seconds=recent_window + previous_window)
    recent_cutoff = now - timedelta(seconds=recent_window)
    
    previous_count = 0
    with history_lock:
        if line_id in crossing_history:
            for direction in ["up", "down", "left", "right"]:
                for timestamp, track_id in crossing_history[line_id][direction]:
                    if previous_cutoff <= timestamp < recent_cutoff:
                        previous_count += 1
    
    previous_rate = previous_count / (previous_window / 60.0) if previous_window > 0 else 0.0
    
    # Calculate percentage change
    if previous_rate > 0:
        pct_change = ((recent_rate - previous_rate) / previous_rate) * 100
    else:
        pct_change = 0.0 if recent_rate == 0 else 100.0
    
    # Determine trend
    if abs(pct_change) < 10:
        return ("Stable", "→", pct_change)
    elif pct_change > 0:
        return ("Increasing", "↑", pct_change)
    else:
        return ("Decreasing", "↓", pct_change)

def get_minute_history(line_id):
    """Get last 60 minutes of bucketed counts"""
    with buckets_lock:
        if line_id in minute_buckets:
            return list(minute_buckets[line_id]["buckets"])
        return []

def calculate_line_stats(line_id, line_data):
    """Calculate all statistics for a line (with caching)"""
    now = datetime.now()
    
    # Check cache
    with stats_cache_lock:
        if line_id in stats_cache:
            cache_entry = stats_cache[line_id]
            if (now - cache_entry["timestamp"]).total_seconds() < STATS_CACHE_SECONDS:
                return cache_entry["stats"]
    
    # Calculate fresh stats
    rate_1min = get_vehicles_per_minute(line_id, 60)
    rate_5min = get_vehicles_per_minute(line_id, 300)
    rate_15min = get_vehicles_per_minute(line_id, 900)
    rate_60min = get_vehicles_per_minute(line_id, 3600)
    
    directional = get_directional_density(line_id, 300)
    
    thresholds = {
        "low": line_data.get("threshold_low", 5.0),
        "normal": line_data.get("threshold_normal", 15.0),
        "high": line_data.get("threshold_high", 25.0)
    }
    level_name, level_color, level_emoji = get_congestion_level(rate_5min, thresholds)
    
    trend_name, trend_emoji, trend_pct = get_trend(line_id, 300, 300)
    
    history = get_minute_history(line_id)
    
    stats = {
        "rates": {
            "current": round(rate_1min, 2),
            "short": round(rate_5min, 2),
            "medium": round(rate_15min, 2),
            "long": round(rate_60min, 2)
        },
        "directional": {
            "up": round(directional["up"], 2),
            "down": round(directional["down"], 2),
            "left": round(directional["left"], 2),
            "right": round(directional["right"], 2)
        },
        "congestion": {
            "level": level_name,
            "color": level_color,
            "emoji": level_emoji
        },
        "trend": {
            "direction": trend_name,
            "emoji": trend_emoji,
            "change_pct": round(trend_pct, 1)
        },
        "history": history
    }
    
    # Cache the result
    with stats_cache_lock:
        stats_cache[line_id] = {
            "stats": stats,
            "timestamp": now
        }
    
    return stats

def point_in_polygon(point, polygon_points):
    """Check if a point is inside a polygon using OpenCV"""
    # Convert polygon points to numpy array format
    polygon = np.array([[p.x, p.y] for p in polygon_points], dtype=np.int32)
    # Use OpenCV's pointPolygonTest (-1: outside, 0: on edge, 1: inside)
    result = cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False)
    return result >= 0

def ccw(A, B, C):
    """Check if three points are in counter-clockwise order"""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def line_segments_intersect(A, B, C, D):
    """Check if line segment AB intersects with line segment CD"""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def get_line_crossing_direction(p1, p2, line_start, line_end):
    """
    Determine the direction of crossing (up/down/left/right)
    Returns: 'up', 'down', 'left', 'right', or None
    """
    # Calculate line vector
    line_vec = (line_end.x - line_start.x, line_end.y - line_start.y)
    # Calculate movement vector
    move_vec = (p2[0] - p1[0], p2[1] - p1[1])
    
    # Cross product to determine which side
    cross = line_vec[0] * move_vec[1] - line_vec[1] * move_vec[0]
    
    # Determine if line is more horizontal or vertical
    line_angle = np.arctan2(line_vec[1], line_vec[0]) * 180 / np.pi
    
    if abs(line_angle) < 45 or abs(line_angle) > 135:
        # Line is more horizontal
        if cross > 0:
            return "down"  # Crossing downward
        else:
            return "up"  # Crossing upward
    else:
        # Line is more vertical
        if cross > 0:
            return "right"  # Crossing to the right
        else:
            return "left"  # Crossing to the left

def check_line_crossing(track_id, current_center, lines, frame_num):
    """
    Check if a vehicle crossed any counting line
    Returns: List of (line_id, direction) tuples for crossed lines
    """
    crossings = []
    
    with positions_lock:
        if track_id not in vehicle_positions:
            vehicle_positions[track_id] = []
        
        # Add current position
        vehicle_positions[track_id].append((current_center[0], current_center[1], frame_num))
        
        # Keep only last 30 frames (about 1 second at 30fps)
        if len(vehicle_positions[track_id]) > 30:
            vehicle_positions[track_id] = vehicle_positions[track_id][-30:]
        
        # Need at least 2 positions to check crossing
        if len(vehicle_positions[track_id]) < 2:
            return crossings
        
        # Get previous position
        prev_pos = vehicle_positions[track_id][-2]
        curr_pos = vehicle_positions[track_id][-1]
        
        # Check against all enabled lines
        for line in lines:
            if not line.enabled:
                continue
            
            line_start = (line.start.x, line.start.y)
            line_end = (line.end.x, line.end.y)
            
            # Check if movement crossed the line
            intersects = line_segments_intersect(
                (prev_pos[0], prev_pos[1]),
                (curr_pos[0], curr_pos[1]),
                line_start,
                line_end
            )
            
            if intersects:
                direction = get_line_crossing_direction(
                    (prev_pos[0], prev_pos[1]),
                    (curr_pos[0], curr_pos[1]),
                    line.start,
                    line.end
                )
                crossings.append((line.id, direction))
                # Debug logging disabled - view counts in frontend UI instead
    
    return crossings

def get_zones_bounding_box(zones, padding=50):
    """Get bounding box that encompasses all enabled zones with padding"""
    if not zones:
        return None
    
    enabled_zones = [z for z in zones if z.enabled]
    if not enabled_zones:
        return None
    
    # Collect all points from all enabled zones
    all_points = []
    for zone in enabled_zones:
        all_points.extend([(p.x, p.y) for p in zone.points])
    
    if not all_points:
        return None
    
    # Find min/max coordinates
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    
    min_x = max(0, int(min(xs)) - padding)
    min_y = max(0, int(min(ys)) - padding)
    max_x = int(max(xs)) + padding
    max_y = int(max(ys)) + padding
    
    return (min_x, min_y, max_x, max_y)

def is_detection_in_zones(bbox, zones, offset_x=0, offset_y=0):
    """Check if detection bounding box center is in any enabled zone
    
    Args:
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        zones: List of Zone objects
        offset_x: X offset if frame was cropped
        offset_y: Y offset if frame was cropped
    """
    if not zones:
        return True  # If no zones defined, detect everywhere
    
    # Calculate center point of bounding box (in original frame coordinates)
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2 + offset_x
    center_y = (y1 + y2) / 2 + offset_y
    
    # Check if center is in any enabled zone
    for zone in zones:
        if zone.enabled and point_in_polygon((center_x, center_y), zone.points):
            return True
    
    return False

def cleanup_all_captures():
    """Force cleanup of all video captures on shutdown"""
    print("\nCleaning up all video captures...")
    with capture_lock:
        for cap in active_captures:
            if cap is not None and cap.isOpened():
                cap.release()
        active_captures.clear()
    print("All captures released.")

# Register cleanup function to run on exit
atexit.register(cleanup_all_captures)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    print("Application starting up...")
    load_zones()  # Load saved zones on startup
    load_lines()  # Load saved counting lines on startup
    yield
    # Shutdown: signal all generators to stop
    print("Application shutting down...")
    shutdown_event.set()
    cleanup_all_captures()
    save_zones()  # Save zones on shutdown
    save_lines()  # Save counting lines on shutdown

app = FastAPI(lifespan=lifespan)

# Allow CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the YOLO model (nano version for speed)
# It will download automatically on first run
model = YOLO('yolo12m.pt') 

# Detection threshold configuration
CONFIDENCE_THRESHOLD = 0.4  # Minimum confidence score (0.0 to 1.0) - adjusted lower for tracking
# When using tracking, you can use a lower threshold (0.4) because the tracker filters false positives
# Without tracking: 0.5-0.6 recommended
# With tracking: 0.3-0.4 recommended (tracker maintains stability)

# Object tracking configuration
USE_TRACKING = True  # Enable object tracking to reduce flickering
# Tracking helps maintain consistent detections across frames even when confidence drops temporarily
# Benefits: Reduces flickering, assigns unique IDs, handles temporary occlusions
TRACKER_TYPE = "bytetrack.yaml"  # Options: "bytetrack.yaml" (faster), "botsort.yaml" (more accurate)

# Class filtering configuration - Only detect specific object classes
# COCO dataset class IDs: person=0, bicycle=1, car=2, motorcycle=3, bus=5, truck=7
DETECT_CLASSES = [0, 2, 3]  # Only detect: person (0), car (2), motorcycle (3)
# To add more classes, add their IDs: [0, 2, 3, 5, 7] = person, car, motorcycle, bus, truck
# Set to None to detect all classes

# ROI Optimization - Process only zone areas for faster inference
USE_ROI_OPTIMIZATION = True  # Set to False to process entire frame always
# When True: YOLO only processes the bounding box containing all zones (50-70% faster!)
# When False: YOLO processes entire frame, then filters detections (slightly more accurate)
# Performance impact: With zones covering 30% of frame, expect 2-3x faster inference

# Visual debugging options
SHOW_ROI_BOX = False  # Show blue rectangle indicating optimized processing area
# Set to True to see what area is being processed for debugging

# RTSP Stream URL with authentication
# Format: rtsp://username:password@ip:port/path
# Replace 'admin' and 'password123' with your actual camera credentials
RTSP_USERNAME = "admin"  # Change this to your camera username
RTSP_PASSWORD = "vivn1213"  # Change this to your camera password
RTSP_URL = f"rtsp://{RTSP_USERNAME}:{RTSP_PASSWORD}@172.16.16.254:554/Streaming/Channels/101?transportmode=unicast&profile=Profile_1"

def generate_frames():
    cap = None
    try:
        cap = cv2.VideoCapture(RTSP_URL)
        
        # Register this capture instance
        with capture_lock:
            active_captures.append(cap)
        
        # Set buffer size to 1 to reduce latency and make read operations more responsive
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Set a timeout for read operations (in milliseconds)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
        
        # Simple retry logic or fallback could be added here
        if not cap.isOpened():
            print(f"Error: Could not open video stream at {RTSP_URL}")
            # For testing purposes without the actual camera, you might want to fallback to 0 (webcam)
            # cap = cv2.VideoCapture(0)
            return  # Exit generator if stream cannot be opened

        retry_count = 0
        max_retries = 3
        frame_count = 0
        
        # # FPS calculation variables
        # session_start_time = time.time()  # Track total session time
        # fps_start_time = time.time()
        # fps_frame_count = 0
        # fps_display_interval = 1.0  # Display FPS every second
        # current_fps = 0.0
        
        print(f"Video stream connected. Starting frame generation...")

        while not shutdown_event.is_set():
            # Check for shutdown at the start of each iteration
            if shutdown_event.is_set():
                print("Shutdown event detected, stopping frame generation")
                break
            
            # Try to read frame with timeout handling
            success, frame = cap.read()
            
            # Check again for shutdown after potentially blocking read
            if shutdown_event.is_set():
                print("Shutdown event detected after frame read")
                break
            
            if not success:
                # Check if we should stop due to shutdown
                if shutdown_event.is_set():
                    break
                    
                # If the stream drops, we try to reconnect
                retry_count += 1
                if retry_count > max_retries or shutdown_event.is_set():
                    print(f"Failed to reconnect after {max_retries} attempts or shutdown requested. Stopping stream.")
                    break
                
                print(f"Stream dropped. Reconnecting... (Attempt {retry_count}/{max_retries})")
                if cap is not None:
                    with capture_lock:
                        if cap in active_captures:
                            active_captures.remove(cap)
                    cap.release()
                    
                # Check for shutdown before reconnecting
                if shutdown_event.is_set():
                    break
                    
                cap = cv2.VideoCapture(RTSP_URL)
                with capture_lock:
                    active_captures.append(cap)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
                time.sleep(0.5)  # Small delay before retrying
                continue
            
            # Reset retry count on successful read
            retry_count = 0
            frame_count += 1
            # fps_frame_count += 1

            # # Calculate and display FPS every interval
            # elapsed_time = time.time() - fps_start_time
            # if elapsed_time >= fps_display_interval:
            #     current_fps = fps_frame_count / elapsed_time
            #     print(f"FPS: {current_fps:.2f} | Total frames: {frame_count}")
            #     # Reset FPS counter
            #     fps_start_time = time.time()
            #     fps_frame_count = 0

            # Get current zones for ROI optimization
            with zones_lock:
                current_zones = list(detection_zones)
            
            # Optimize inference by cropping to zone bounding box (if enabled)
            roi_bbox = None
            if USE_ROI_OPTIMIZATION and current_zones:
                roi_bbox = get_zones_bounding_box(current_zones, padding=50)
            
            if roi_bbox:
                # OPTIMIZATION: Process only the region containing all zones
                min_x, min_y, max_x, max_y = roi_bbox
                # Ensure coordinates are within frame bounds
                frame_h, frame_w = frame.shape[:2]
                min_x = max(0, min_x)
                min_y = max(0, min_y)
                max_x = min(frame_w, max_x)
                max_y = min(frame_h, max_y)
                
                # Crop frame to ROI
                cropped_frame = frame[min_y:max_y, min_x:max_x]
                
                # Run inference on cropped region only (FASTER!)
                if USE_TRACKING:
                    results = model.track(
                        cropped_frame, 
                        verbose=False, 
                        conf=CONFIDENCE_THRESHOLD,
                        classes=DETECT_CLASSES,
                        tracker=TRACKER_TYPE,
                        persist=True
                    )
                else:
                    results = model(cropped_frame, verbose=False, conf=CONFIDENCE_THRESHOLD, classes=DETECT_CLASSES)
                
                # Map detections back to original frame coordinates
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    # Clone boxes data to avoid in-place modification error
                    adjusted_boxes = []
                    for box in results[0].boxes:
                        # Clone the box data
                        box_data = box.data.clone()
                        # Adjust coordinates to original frame (x1, y1, x2, y2 are first 4 values)
                        box_data[0][0] += min_x  # x1
                        box_data[0][1] += min_y  # y1
                        box_data[0][2] += min_x  # x2
                        box_data[0][3] += min_y  # y2
                        adjusted_boxes.append(box_data)
                    
                    # Update orig_shape to full frame
                    results[0].orig_shape = frame.shape[:2]
                    
                    # Replace boxes with adjusted coordinates
                    if adjusted_boxes:
                        results[0].boxes = type(results[0].boxes)(
                            torch.cat(adjusted_boxes),
                            results[0].orig_shape
                        )
                
                # Filter detections by precise polygon boundaries
                if results[0].boxes is not None:
                    filtered_boxes = []
                    for box in results[0].boxes:
                        bbox = box.xyxy[0].cpu().numpy()
                        # Note: offset is 0 because we already adjusted coordinates
                        if is_detection_in_zones(bbox, current_zones, offset_x=0, offset_y=0):
                            filtered_boxes.append(box.data.clone())
                    
                    if filtered_boxes:
                        results[0].boxes = type(results[0].boxes)(
                            torch.cat(filtered_boxes),
                            results[0].orig_shape
                        )
                    else:
                        results[0].boxes = None
            else:
                # NO ZONES: Process entire frame (standard behavior)
                if USE_TRACKING:
                    results = model.track(
                        frame, 
                        verbose=False, 
                        conf=CONFIDENCE_THRESHOLD,
                        classes=DETECT_CLASSES,
                        tracker=TRACKER_TYPE,
                        persist=True
                    )
                else:
                    results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD, classes=DETECT_CLASSES)
            
            # Plot the results on the original frame (draws bounding boxes and track IDs)
            # Note: results[0].plot() uses orig_shape which we set to full frame
            annotated_frame = results[0].plot(img=frame)
            
            # Get current counting lines
            with lines_lock:
                current_lines = list(counting_lines)
            
            # Debug logging disabled - view counts in frontend UI instead
            # if frame_count % 100 == 0 and current_lines:
            #     print(f"📊 Active counting lines: {len(current_lines)}")
            
            # Check for line crossings (only if tracking is enabled)
            if USE_TRACKING and current_lines and results[0].boxes is not None:
                for box in results[0].boxes:
                    # Get track ID (if tracking is enabled, boxes have track IDs)
                    track_id = None
                    
                    # Try different ways to access track ID
                    if hasattr(box, 'id') and box.id is not None:
                        try:
                            track_id = int(box.id.item() if hasattr(box.id, 'item') else box.id)
                        except:
                            pass
                    
                    # Fallback: check if ID is in the data tensor
                    if track_id is None and hasattr(box, 'data'):
                        # In tracking mode, data format is [x1, y1, x2, y2, track_id, conf, cls]
                        box_data = box.data[0].cpu().numpy()
                        if len(box_data) >= 7:  # Has track_id
                            track_id = int(box_data[4])
                    
                    if track_id is not None:
                        # Get bounding box center
                        bbox = box.xyxy[0].cpu().numpy()
                        center_x = (bbox[0] + bbox[2]) / 2
                        center_y = (bbox[1] + bbox[3]) / 2
                        
                        # Check for line crossings
                        crossings = check_line_crossing(track_id, (center_x, center_y), current_lines, frame_count)
                        
                        # Record crossings with timestamps
                        for line_id, direction in crossings:
                            now = datetime.now()
                            
                            # Add to timestamp history
                            with history_lock:
                                if line_id not in crossing_history:
                                    initialize_line_data_structures(line_id)
                                crossing_history[line_id][direction].append((now, track_id))
                            
                            # Update minute bucket
                            with buckets_lock:
                                if line_id not in minute_buckets:
                                    initialize_line_data_structures(line_id)
                                
                                update_minute_buckets_internal(line_id)
                                minute_buckets[line_id]["current_count"] += 1
                            
                            # Logging disabled - view counts in frontend UI instead
            
            # Periodically clean old data (every 1000 frames)
            if frame_count % 1000 == 0:
                clean_old_crossing_data()
            
            # Draw detection zones on the frame
            for zone in current_zones:
                if zone.enabled:
                    # Convert points to numpy array
                    pts = np.array([[int(p.x), int(p.y)] for p in zone.points], dtype=np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    
                    # Parse color (hex to BGR)
                    try:
                        hex_color = zone.color.lstrip('#')
                        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                        color = (b, g, r)  # OpenCV uses BGR
                    except:
                        color = (0, 255, 0)  # Default green
                    
                    # Draw filled polygon with transparency
                    overlay = annotated_frame.copy()
                    cv2.fillPoly(overlay, [pts], color)
                    cv2.addWeighted(overlay, 0.15, annotated_frame, 0.85, 0, annotated_frame)
                    
                    # Draw polygon border
                    cv2.polylines(annotated_frame, [pts], True, color, 3)
                    
                    # Draw zone name
                    if pts.shape[0] > 0:
                        text_pos = (int(pts[0][0][0]), int(pts[0][0][1]) - 10)
                        cv2.putText(annotated_frame, zone.name, text_pos,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw ROI bounding box (blue rectangle showing optimized processing area)
            # Only shown if SHOW_ROI_BOX is True (for debugging)
            if roi_bbox and SHOW_ROI_BOX:
                min_x, min_y, max_x, max_y = roi_bbox
                frame_h, frame_w = frame.shape[:2]
                min_x = max(0, min_x)
                min_y = max(0, min_y)
                max_x = min(frame_w, max_x)
                max_y = min(frame_h, max_y)
                
                # Draw rectangle to show optimized ROI (for debugging)
                cv2.rectangle(annotated_frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
                cv2.putText(annotated_frame, "Optimized ROI", (min_x + 5, min_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Draw counting lines
            for line in current_lines:
                if line.enabled:
                    # Parse color (hex to BGR)
                    try:
                        hex_color = line.color.lstrip('#')
                        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                        color = (b, g, r)  # OpenCV uses BGR
                    except:
                        color = (0, 0, 255)  # Default red
                    
                    start_pt = (int(line.start.x), int(line.start.y))
                    end_pt = (int(line.end.x), int(line.end.y))
                    
                    # Draw thick line
                    cv2.line(annotated_frame, start_pt, end_pt, color, 4)
                    
                    # Draw directional arrows
                    arrow_length = 30
                    line_vec = np.array([end_pt[0] - start_pt[0], end_pt[1] - start_pt[1]])
                    line_len = np.linalg.norm(line_vec)
                    if line_len > 0:
                        line_vec = line_vec / line_len
                    
                    # Perpendicular vector (for up/down indicators)
                    perp_vec = np.array([-line_vec[1], line_vec[0]])
                    
                    # Draw arrows on both sides of line
                    mid_pt = ((start_pt[0] + end_pt[0]) // 2, (start_pt[1] + end_pt[1]) // 2)
                    
                    # Arrow 1 (one direction)
                    arrow1_end = (int(mid_pt[0] + perp_vec[0] * arrow_length), 
                                  int(mid_pt[1] + perp_vec[1] * arrow_length))
                    cv2.arrowedLine(annotated_frame, mid_pt, arrow1_end, color, 2, tipLength=0.5)
                    
                    # Arrow 2 (opposite direction)
                    arrow2_end = (int(mid_pt[0] - perp_vec[0] * arrow_length), 
                                  int(mid_pt[1] - perp_vec[1] * arrow_length))
                    cv2.arrowedLine(annotated_frame, mid_pt, arrow2_end, color, 2, tipLength=0.5)
                    
                    # Counting text removed - view counts in frontend UI for cleaner look
                    # counts = crossing_counts.get(line.id, {"up": 0, "down": 0, "left": 0, "right": 0, "total": 0})
                    # count_text = f"{line.name}: ↑{counts['up']} ↓{counts['down']} Total:{counts['total']}"
                    # cv2.putText(annotated_frame, count_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Encode the frame to JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()

            # Check one more time before yielding
            if shutdown_event.is_set():
                break

            # Yield the frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    except GeneratorExit:
        print("Client disconnected from video stream")
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received in generate_frames")
    except Exception as e:
        if not shutdown_event.is_set():
            print(f"Error in generate_frames: {e}")
    finally:
        # Always clean up resources
        if cap is not None:
            with capture_lock:
                if cap in active_captures:
                    active_captures.remove(cap)
            try:
                cap.release()
                # # Calculate average FPS for the entire session
                # if 'session_start_time' in locals() and 'frame_count' in locals():
                #     total_time = time.time() - session_start_time
                #     avg_fps = frame_count / total_time if total_time > 0 else 0
                #     print(f"Video capture released. Total frames: {frame_count}, Session duration: {total_time:.2f}s, Average FPS: {avg_fps:.2f}")
                # else:
                #     print(f"Video capture released.")
                print(f"Video capture released. Total frames: {frame_count}")
            except Exception as e:
                print(f"Video capture released.")

@app.get("/")
def read_root():
    return {"message": "YOLO Object Detection API is running"}

@app.get("/video_feed")
def video_feed():
    """
    Returns a streaming response using MJPEG protocol.
    The frontend can display this directly in an <img> tag.
    """
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# Zone Management API Endpoints

@app.get("/api/zones")
def get_zones():
    """Get all detection zones"""
    with zones_lock:
        return {"zones": [zone.model_dump() for zone in detection_zones]}

@app.post("/api/zones")
def create_zone(zone: Zone):
    """Create a new detection zone"""
    with zones_lock:
        # Check if zone with same ID exists
        existing_ids = [z.id for z in detection_zones]
        if zone.id in existing_ids:
            return JSONResponse(
                status_code=400,
                content={"error": f"Zone with ID '{zone.id}' already exists"}
            )
        detection_zones.append(zone)
    save_zones()
    return {"message": "Zone created successfully", "zone": zone.model_dump()}

@app.put("/api/zones/{zone_id}")
def update_zone(zone_id: str, zone: Zone):
    """Update an existing detection zone"""
    with zones_lock:
        for i, z in enumerate(detection_zones):
            if z.id == zone_id:
                detection_zones[i] = zone
                save_zones()
                return {"message": "Zone updated successfully", "zone": zone.model_dump()}
    return JSONResponse(
        status_code=404,
        content={"error": f"Zone with ID '{zone_id}' not found"}
    )

@app.delete("/api/zones/{zone_id}")
def delete_zone(zone_id: str):
    """Delete a detection zone"""
    with zones_lock:
        for i, z in enumerate(detection_zones):
            if z.id == zone_id:
                deleted_zone = detection_zones.pop(i)
                save_zones()
                return {"message": "Zone deleted successfully", "zone": deleted_zone.model_dump()}
    return JSONResponse(
        status_code=404,
        content={"error": f"Zone with ID '{zone_id}' not found"}
    )

@app.delete("/api/zones")
def delete_all_zones():
    """Delete all detection zones"""
    with zones_lock:
        count = len(detection_zones)
        detection_zones.clear()
    save_zones()
    return {"message": f"Deleted {count} zones"}

# Counting Lines API Endpoints

@app.get("/api/lines")
def get_lines():
    """Get all counting lines with their density statistics (cached)"""
    with lines_lock:
        lines_data = [line.model_dump() for line in counting_lines]
    
    # Add cached density statistics to each line
    for line_data in lines_data:
        line_id = line_data["id"]
        line_data["stats"] = calculate_line_stats(line_id, line_data)
    
    return {"lines": lines_data}

@app.post("/api/lines")
def create_line(line: CountingLine):
    """Create a new counting line"""
    with lines_lock:
        # Check if line with same ID exists
        existing_ids = [l.id for l in counting_lines]
        if line.id in existing_ids:
            return JSONResponse(
                status_code=400,
                content={"error": f"Line with ID '{line.id}' already exists"}
            )
        counting_lines.append(line)
    # Initialize data structures for this line
    initialize_line_data_structures(line.id)
    save_lines()
    return {"message": "Counting line created successfully", "line": line.model_dump()}

@app.put("/api/lines/{line_id}")
def update_line(line_id: str, line: CountingLine):
    """Update an existing counting line"""
    with lines_lock:
        for i, l in enumerate(counting_lines):
            if l.id == line_id:
                counting_lines[i] = line
                save_lines()
                return {"message": "Line updated successfully", "line": line.model_dump()}
    return JSONResponse(
        status_code=404,
        content={"error": f"Line with ID '{line_id}' not found"}
    )

@app.delete("/api/lines/{line_id}")
def delete_line(line_id: str):
    """Delete a counting line"""
    deleted_line = None
    
    # First, remove from counting_lines list
    with lines_lock:
        for i, l in enumerate(counting_lines):
            if l.id == line_id:
                deleted_line = counting_lines.pop(i)
                break
    
    # If line not found, return error
    if deleted_line is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"Line with ID '{line_id}' not found"}
        )
    
    # Now remove all associated data (without holding lines_lock)
    # This prevents deadlock with the video generation thread
    with history_lock:
        if line_id in crossing_history:
            del crossing_history[line_id]
    
    with buckets_lock:
        if line_id in minute_buckets:
            del minute_buckets[line_id]
    
    with stats_cache_lock:
        if line_id in stats_cache:
            del stats_cache[line_id]
    
    save_lines()
    return {"message": "Line deleted successfully", "line": deleted_line.model_dump()}

@app.delete("/api/lines")
def delete_all_lines():
    """Delete all counting lines"""
    # First, clear the counting_lines list
    with lines_lock:
        count = len(counting_lines)
        counting_lines.clear()
    
    # Now clear all associated data (without holding lines_lock)
    # This prevents deadlock with the video generation thread
    with history_lock:
        crossing_history.clear()
    
    with buckets_lock:
        minute_buckets.clear()
    
    with stats_cache_lock:
        stats_cache.clear()
    
    save_lines()
    return {"message": f"Deleted {count} lines"}

@app.post("/api/lines/{line_id}/reset")
def reset_line_counts(line_id: str):
    """Reset all crossing data for a specific line"""
    with lines_lock:
        if not any(l.id == line_id for l in counting_lines):
            return JSONResponse(
                status_code=404,
                content={"error": f"Line with ID '{line_id}' not found"}
            )
    
    # Clear all historical data
    with history_lock:
        if line_id in crossing_history:
            for direction in crossing_history[line_id]:
                crossing_history[line_id][direction].clear()
    
    with buckets_lock:
        if line_id in minute_buckets:
            minute_buckets[line_id]["buckets"].clear()
            minute_buckets[line_id]["current_count"] = 0
            minute_buckets[line_id]["current_minute"] = datetime.now().replace(second=0, microsecond=0)
    
    # Clear stats cache
    with stats_cache_lock:
        if line_id in stats_cache:
            del stats_cache[line_id]
    
    return {"message": f"Data reset for line {line_id}"}

@app.post("/api/lines/reset-all")
def reset_all_counts():
    """Reset all line crossing data"""
    with history_lock:
        for line_id in crossing_history:
            for direction in crossing_history[line_id]:
                crossing_history[line_id][direction].clear()
    
    with buckets_lock:
        for line_id in minute_buckets:
            minute_buckets[line_id]["buckets"].clear()
            minute_buckets[line_id]["current_count"] = 0
            minute_buckets[line_id]["current_minute"] = datetime.now().replace(second=0, microsecond=0)
    
    # Clear all stats cache
    with stats_cache_lock:
        stats_cache.clear()
    
    return {"message": "All counts reset"}

if __name__ == "__main__":
    import uvicorn
    import signal
    import sys
    
    def signal_handler(sig, frame):
        """Handle Ctrl+C signal"""
        print("\n\nCtrl+C detected! Forcing shutdown...")
        shutdown_event.set()
        cleanup_all_captures()
        # Give a brief moment for cleanup
        time.sleep(0.5)
        print("Shutdown complete. Exiting...")
        sys.exit(0)
    
    # Register signal handler for Windows
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        print("Starting YOLO Object Detection API server...")
        print("Press Ctrl+C to stop the server")
        # Run the server with shorter timeout for graceful shutdown
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8000,
            log_level="info",
            timeout_graceful_shutdown=2  # Only wait 2 seconds for connections to close
        )
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")
        shutdown_event.set()
        cleanup_all_captures()
    except Exception as e:
        print(f"Server error: {e}")
        shutdown_event.set()
        cleanup_all_captures()
    finally:
        print("Server stopped.")

