# 🔥 Fire + 🦜 Animal Detection Flask App + Poacher Detection

from flask import Flask, Response, jsonify, render_template_string
from ultralytics import YOLO
import cv2
import time
import random
import requests
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import Counter

app = Flask(__name__)

# YOLO Models
fire_model = YOLO("best.pt")
animal_model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

# 🔥 Fire Detection Settings
conf_threshold = 0.6
iou_threshold = 0.6
log_file = "fire_history.csv"
fire_cooldown_seconds = 10
last_fire_log_time = 0
fire_active = False

# 📸 Snapshot Settings
snapshot_dir = "fire_snapshots"
if not os.path.exists(snapshot_dir):
    os.makedirs(snapshot_dir)

#  Animal Detection
ANIMAL_CLASSES = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
species_counts = {}

# Ensure log file exists
if not os.path.exists(log_file):
    pd.DataFrame(columns=["timestamp", "fire_detected", "latitude", "longitude"]).to_csv(log_file, index=False)

# ✅ Updated location fetch function using IP-API
def get_location():
    try:
        response = requests.get("http://ip-api.com/json/").json()
        lat, lon = response.get("lat"), response.get("lon")
        print(f"📍 Location fetched: {lat}, {lon}")
        return lat, lon
    except Exception as e:
        print("❌ Error fetching location:", e)
        return None, None

def log_fire_event(frame=None, results=None):
    lat, lon = get_location()
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    if not lat or not lon:
        print("⚠️ Fire detected but location unavailable.")
        return
    
    # Save snapshot if frame is provided
    if frame is not None and results is not None:
        # Draw bounding boxes on the frame copy
        frame_with_boxes = frame.copy()
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame_with_boxes, f"Fire {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Add timestamp overlay
        cv2.putText(frame_with_boxes, timestamp, (10, frame_with_boxes.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add snapshot notification text
        notification_text = "Snapshot Captured!"
        cv2.putText(frame_with_boxes, notification_text, (frame_with_boxes.shape[1]//2 - 100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        snapshot_path = os.path.join(snapshot_dir, f"fire_{timestamp.replace(':', '-')}.jpg")
        cv2.imwrite(snapshot_path, frame_with_boxes)
        print(f"📸 Fire snapshot saved with detection boxes: {snapshot_path}")
    
    df = pd.read_csv(log_file)
    df = pd.concat([df, pd.DataFrame([{
        "timestamp": timestamp,
        "fire_detected": True,
        "latitude": lat,
        "longitude": lon
    }])], ignore_index=True)
    df.to_csv(log_file, index=False)

def fire_frames():
    global last_fire_log_time, fire_active
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Create a copy of the frame for processing
        process_frame = frame.copy()
        
        results = fire_model.predict(process_frame, conf=conf_threshold, iou=iou_threshold, imgsz=640, verbose=False)
        for result in results:
            result.names = {0: "fire"}
            annotated_frame = result.plot()

        fire_detected = len(results[0].boxes) > 0
        current_time = time.time()

        if fire_detected:
            border_color = (0, 0, 255) if int(current_time % 1 < 0.5) else (255, 0, 0)
            thickness = 20
            h, w, _ = annotated_frame.shape
            cv2.rectangle(annotated_frame, (0, 0), (w, h), border_color, thickness)
            if not fire_active or (current_time - last_fire_log_time) >= fire_cooldown_seconds:
                log_fire_event(frame, results)
                last_fire_log_time = current_time
                fire_active = True
        else:
            fire_active = False

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def animal_frames():
    global species_counts
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Create a copy of the frame for processing
        process_frame = frame.copy()
        
        results = animal_model(process_frame, verbose=False)[0]
        detected = []
        for box in results.boxes:
            label = animal_model.names[int(box.cls[0])]
            if label in ANIMAL_CLASSES:
                detected.append(label)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        species_counts = dict(Counter(detected))
        y = 25
        cv2.putText(frame, f"Total Animals: {len(detected)}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        y += 30
        for s, c in species_counts.items():
            cv2.putText(frame, f"{s.capitalize()}: {c}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            y += 25

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def poacher_frames():
    try:
        with open("class.txt", "r") as f:
            weapon_classes = f.read().split("\n")
    except FileNotFoundError:
        print("Error: class.txt not found.")
        return

    # Ensure poaching_snapshots directory exists
    snapshot_dir = "poaching_snapshots"
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)

    person_model = YOLO("yolov8n.pt")
    weapon_model = YOLO("yo.pt")
    weapon_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in weapon_classes]
    poacher_weapon_ids = [0, 1, 2, 3, 4, 5, 7]  # Customize as needed

    tracker = None
    tracking = False
    last_snapshot_time = 0
    snapshot_cooldown = 5  # Seconds between snapshots

    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Create a copy of the frame for processing
        process_frame = frame.copy()

        if tracking and tracker is not None:
            success_track, box = tracker.update(frame)
            if success_track:
                x, y, w, h = map(int, box)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Tracking Poacher", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                tracking = False
                tracker = None

        # Detect persons
        person_results = person_model.predict(source=[frame], conf=0.6, save=False)
        person_boxes = person_results[0].boxes
        person_detected = []

        if person_boxes is not None:
            for box in person_boxes:
                cls_id = int(box.cls.cpu().numpy()[0])
                if cls_id != 0:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
                conf = box.conf.cpu().numpy()[0]
                person_detected.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {conf*100:.1f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Detect weapons
        weapon_results = weapon_model.predict(source=[frame], conf=0.6, save=False)
        weapon_boxes = weapon_results[0].boxes
        poacher_detected = False

        if weapon_boxes is not None:
            for box in weapon_boxes:
                cls_id = int(box.cls.cpu().numpy()[0])
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
                conf = box.conf.cpu().numpy()[0]
                color = weapon_colors[cls_id % len(weapon_colors)]
                label = f"{weapon_classes[cls_id]} {conf*100:.1f}%"

                if cls_id in poacher_weapon_ids:
                    poacher_detected = True

                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Start tracking if poacher found
        if poacher_detected and not tracking:
            for (x1, y1, x2, y2) in person_detected:
                bbox = (x1, y1, x2 - x1, y2 - y1)
                tracker = cv2.TrackerCSRT_create()
                tracking = tracker.init(frame, bbox)
                break

        if poacher_detected:
            cv2.putText(frame, "Poacher Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            # Save snapshot with timestamp
            current_time = time.time()
            if current_time - last_snapshot_time >= snapshot_cooldown:
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                # Add timestamp overlay
                frame_with_timestamp = frame.copy()
                cv2.putText(frame_with_timestamp, timestamp, (10, frame_with_timestamp.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                # Add snapshot notification text
                notification_text = "Snapshot Captured!"
                cv2.putText(frame_with_timestamp, notification_text, (frame_with_timestamp.shape[1]//2 - 100, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                snapshot_path = os.path.join(snapshot_dir, f"poacher_{timestamp.replace(':', '-')}.jpg")
                cv2.imwrite(snapshot_path, frame_with_timestamp)
                print(f"📸 Poacher snapshot saved: {snapshot_path}")
                last_snapshot_time = current_time

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# Routes
@app.route('/')
def index():
        return '''
    <h2>🔥 Fire & 🦜 Animal & 🧍‍♂️ Poacher Detection Dashboard</h2>
    <ul>
        <li><a href="/video_feed">Fire Detection</a></li>
        <li><a href="/video_species">Animal Detection</a></li>
        <li><a href="/video_poacher">Poacher Detection</a></li>
        <li><a href="/species_data">Species Data API</a></li>
        <li><a href="/history">Fire History</a></li>
        <li><a href="/map">Fire Map</a></li>
    </ul>
    '''


@app.route('/video_feed')
def video_feed():
    return Response(fire_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/video_species')
def video_species():
    return Response(animal_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/species_data')
def species_data():
    return jsonify(species_counts)

@app.route('/video_poacher')
def video_poacher():
    return Response(poacher_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/history')
def history():
    df = pd.read_csv(log_file)
    plt.figure(figsize=(10,4))
    times = pd.to_datetime(df["timestamp"])
    df["count"] = df["fire_detected"].astype(int)
    plt.plot(times, df["count"], marker='o')
    plt.title("🔥 Fire Detection Over Time")
    plt.xlabel("Time")
    plt.ylabel("Detections")
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs("static", exist_ok=True)
    plt.savefig("static/fire_chart.png")
    plt.close()
    return render_template_string('<h2>🔥 Fire History</h2><img src="/static/fire_chart.png"><br><a href="/">Back</a>')

@app.route('/map')
def fire_map():
    df = pd.read_csv(log_file)
    pins = ""
    for _, r in df.iterrows():
        if r["fire_detected"] and pd.notnull(r["latitude"]) and pd.notnull(r["longitude"]):
            pins += f"L.marker([{r['latitude']}, {r['longitude']}]).addTo(map).bindPopup('🔥 {r['timestamp']}');\n"

    return render_template_string(f'''
    <html><head>
    <link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
    <style>#map {{ height: 600px; }}</style></head>
    <body><h2>🔥 Fire Map</h2>
    <div id="map"></div>
    <script>
        var map = L.map('map').setView([20.6, 78.9], 5);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap'
        }}).addTo(map);
        {pins}
    </script>
    <a href="/">Back</a></body></html>
    ''')

if __name__ == '__main__':
    app.run(debug=True)
