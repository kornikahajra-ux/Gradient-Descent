from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from geopy.distance import geodesic
import datetime

# Initialize the API
app = FastAPI()

# Allow the frontend HTML to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your perfectly trained ML model and the NGO tasks
model = joblib.load('volunteer_matcher.pkl')
tasks_df = pd.read_csv('synthetic_tasks_demo.csv') # Make sure this matches your filename!

# Define what data the frontend will send us
class VolunteerProfile(BaseModel):
    name: str
    skills: list[str]
    lat: float
    lon: float
    availability: str

@app.post("/recommend")
async def recommend(vol: VolunteerProfile):
    recommendations = []
    
    for _, task in tasks_df.iterrows():
        # 1. Geographic Distance
        dist = geodesic((vol.lat, vol.lon), (task['lat'], task['lon'])).km
        
        # 2. Skill Match
        skill_match = 1 if task['required_skill'] in vol.skills else 0
        
        # 3. Time Logic
        task_time = pd.to_datetime(task['timestamp'])
        is_emergency = task['urgency'] >= 4
        is_working_hour = 9 <= task_time.hour <= 17
        
        time_score = 0
        if vol.availability == 'Full-time' or is_emergency:
            time_score = 1
        elif vol.availability == 'Weekday' and is_working_hour:
            time_score = 1
        elif vol.availability == 'Weekend' and task_time.hour > 17:
            time_score = 1
            
        # 4. Prepare features exactly how the model was trained
        features = [[dist, skill_match, time_score, task['urgency']]]
        
        # 5. Predict Match Probability using the loaded model
        # predict_proba returns [[prob_0, prob_1]]. We want prob_1 (Good Match)
        prob = model.predict_proba(features)[0][1]
        
        # Convert numpy values to standard Python floats to avoid the JSON error
        recommendations.append({
            "ngo_name": task['ngo_name'],
            "skill": task['required_skill'],
            "distance": float(round(dist, 1)), # Wrapped in float()
            "match_score": float(round(prob * 100, 1)), # Wrapped in float()
            "urgency": int(task['urgency'])
        })

    # Sort by highest match score, return the top 5
    top_matches = sorted(recommendations, key=lambda x: x['match_score'], reverse=True)[:5]
    return top_matches

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)