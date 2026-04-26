import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from geopy.distance import geodesic
from datetime import datetime
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SKILL_URGENCY_MAP = {
    'Surgeon': 5, 'Nurse': 5, 'EMT': 5, 'First Aid': 5, 'Trauma Care': 5,
    'Search & Rescue': 4, 'Logistics': 4, 'Heavy Lifting': 4, 'Shelter Management': 4,
    'Teaching': 3, 'Mentoring': 3, 'Translation': 3, 'Child Care': 3,
    'IT Support': 2, 'Data Entry': 2, 'Accounting': 2, 'Social Media': 2,
    'Cooking': 1, 'Gardening': 1, 'Cleaning': 1, 'Event Setup': 1
}

# 1. LOAD MODEL
model = joblib.load('volunteer_matcher.pkl')

# 2. INITIALIZE LIVE DATABASE — seed from CSV if it exists
LIVE_CSV = 'live_deployment.csv'
SEED_CSV = 'synthetic_tasks_demo.csv'   # fallback demo data

active_volunteers = {}

def _load_tasks_from_csv(path: str) -> list:
    """Read a CSV into the tasks_db list format. Skips missing files gracefully."""
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
        # Ensure every expected column exists (older CSVs may lack 'status'/'timestamp')
        for col, default in [('status', 'Open'), ('timestamp', '')]:
            if col not in df.columns:
                df[col] = default
        records = df.to_dict(orient='records')
        print(f"[startup] Loaded {len(records)} tasks from '{path}'")
        return records
    except Exception as e:
        print(f"[startup] Could not load '{path}': {e}")
        return []

# Try live state first; fall back to demo seed data
tasks_db: list = _load_tasks_from_csv(LIVE_CSV) or _load_tasks_from_csv(SEED_CSV)

def sync_to_disk():
    """Saves the current in-memory state back to the live CSV."""
    if tasks_db:
        pd.DataFrame(tasks_db).to_csv(LIVE_CSV, index=False)

class NGORequirement(BaseModel):
    ngo_name: str
    required_skill: str
    urgency: int = None
    lat: float
    lon: float

class VolunteerProfile(BaseModel):
    name: str
    skills: list[str]
    lat: float
    lon: float
    availability: str 

@app.post("/ngo/create")
async def create_task(req: NGORequirement):
    new_id = f"TASK_{uuid.uuid4().hex[:6].upper()}"
    final_urgency = req.urgency if req.urgency else SKILL_URGENCY_MAP.get(req.required_skill, 3)
    
    new_task = {
        "task_id": new_id, 
        "ngo_name": req.ngo_name, 
        "required_skill": req.required_skill,
        "urgency": final_urgency, 
        "lat": req.lat, 
        "lon": req.lon, 
        "status": "Open",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Real-time stamp
    }
    tasks_db.insert(0, new_task)
    sync_to_disk()
    return {"message": f"Requirement posted with Urgency Level: {final_urgency}", "task_id": new_id}

@app.post("/recommend")
async def recommend(vol: VolunteerProfile):
    if active_volunteers.get(vol.name) == "Engaged":
        return []

    # Get current real-time clock for the demo
    now = datetime.now()
    current_hour = now.hour
    is_weekend = now.weekday() >= 5 

    recommendations = []
    for task in tasks_db:
        if task.get('status') != 'Open': continue

        # REAL-TIME SHIFT LOGIC
        on_duty = False
        if vol.availability == 'Full-time':
            on_duty = True
        elif vol.availability == 'Weekday':
            if not is_weekend and 9 <= current_hour <= 17:
                on_duty = True
        elif vol.availability == 'Weekend':
            if is_weekend or current_hour > 17 or current_hour < 9:
                on_duty = True

        # Emergency Override (Always show level 5)
        if task['urgency'] >= 5:
            on_duty = True

        if not on_duty: continue

        dist = geodesic((vol.lat, vol.lon), (task['lat'], task['lon'])).km
        skill_match = 1 if task['required_skill'] in vol.skills else 0
        time_score = 1 if on_duty else 0
        
        features = [[dist, skill_match, time_score, task['urgency']]]
        prob = model.predict_proba(features)[0][1]
        
        recommendations.append({
            "task_id": str(task['task_id']),
            "ngo_name": task['ngo_name'],
            "skill": task['required_skill'],
            "distance": float(round(dist, 1)),
            "match_score": float(round(prob * 100, 1)),
            "urgency": int(task['urgency']),
            "lat": task['lat'],
            "lon": task['lon']
        })
        
    return sorted(recommendations, key=lambda x: x['match_score'], reverse=True)[:5]

@app.post("/volunteer/action")
async def volunteer_action(name: str, task_id: str, action: str):
    for task in tasks_db:
        if str(task.get('task_id')) == task_id:
            if action == "accept":
                task['status'] = 'In Progress'
                active_volunteers[name] = "Engaged"
                sync_to_disk()
                return {"message": "Task accepted!"}
            elif action == "finish":
                task['status'] = 'Completed'
                active_volunteers[name] = "Available"
                sync_to_disk()
                return {"message": "You are now Available."}
    return {"message": "Task ID not found"}

@app.get("/volunteer/status/{name}")
async def get_volunteer_status(name: str):
    status = active_volunteers.get(name, "Available")
    current_task = None
    if status == "Engaged":
        for task in tasks_db:
            if task.get('status') == 'In Progress':
                current_task = task.get('task_id')
                break
    return {"status": status, "task_id": current_task}

@app.post("/ngo/update")
async def ngo_update(task_id: str, status: str):
    for task in tasks_db:
        if str(task.get('task_id')) == task_id:
            task['status'] = status
            sync_to_disk()
            return {"message": f"Task {task_id} updated to {status}"}
    raise HTTPException(status_code=404, detail="Task not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)