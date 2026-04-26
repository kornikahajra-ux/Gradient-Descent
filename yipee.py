import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()

# Configuration
NUM_VOLUNTEERS = 500
NUM_TASKS = 200

SKILL_WEIGHTS = {
    'Surgeon': 5, 'Nurse': 5, 'EMT': 5, 'First Aid': 5, 'Trauma Care': 5,
    'Search & Rescue': 4, 'Logistics': 4, 'Heavy Lifting': 4, 'Shelter Management': 4,
    'Teaching': 3, 'Mentoring': 3, 'Translation': 3, 'Child Care': 3,
    'IT Support': 2, 'Data Entry': 2, 'Accounting': 2, 'Social Media': 2,
    'Cooking': 1, 'Gardening': 1, 'Cleaning': 1, 'Event Setup': 1
}

SKILLS = list(SKILL_WEIGHTS.keys())

# --- Helper Function for Time Generation ---
def generate_task_time(skill):
    # Skills that can happen ANYTIME (Medical/Emergency)
    emergency_skills = ['Surgeon', 'Nurse', 'EMT', 'First Aid', 'Trauma Care', 'Search & Rescue']
    
    # Generate a random date within the last 30 days
    base_date = fake.date_time_between(start_date='-30d', end_date='now')
    
    if skill in emergency_skills:
        # Any hour (0-23)
        return base_date
    else:
        # Standard hours: 9 AM to 5 PM (9 to 17)
        return base_date.replace(hour=random.randint(9, 17), minute=random.randint(0, 59))

# 1. Generate Synthetic Volunteers (No changes needed here)
volunteers = []
for i in range(NUM_VOLUNTEERS):
    volunteers.append({
        'vol_id': f'VOL_{i:03d}',
        'name': fake.name(),
        'skill_set': random.sample(SKILLS, k=random.randint(1, 3)),
        'lat': float(fake.coordinate(center=28.6139, radius=0.1)), 
        'lon': float(fake.coordinate(center=77.2090, radius=0.1)),
        'availability': random.choice(['Weekend', 'Weekday', 'Evening', 'Full-time'])
    })

df_volunteers = pd.DataFrame(volunteers)

# 2. Generate Synthetic NGO Tasks WITH Timestamps
tasks = []
for i in range(NUM_TASKS):
    selected_skill = random.choice(SKILLS)
    timestamp = generate_task_time(selected_skill)
    
    tasks.append({
        'task_id': f'TASK_{i:03d}',
        'ngo_name': fake.company(),
        'required_skill': selected_skill,
        'urgency': SKILL_WEIGHTS[selected_skill], 
        'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        'hour_of_day': timestamp.hour, # Useful for ML training
        'lat': float(fake.coordinate(center=28.6139, radius=0.1)),
        'lon': float(fake.coordinate(center=77.2090, radius=0.1))
    })

df_tasks = pd.DataFrame(tasks)

# Preview results
print("--- NGO Task Data Preview (Check the timestamps!) ---")
print(df_tasks[['task_id', 'required_skill', 'timestamp', 'urgency']].head(10))

df_volunteers.to_csv('synthetic_volunteers_demo.csv', index=False)
df_tasks.to_csv('synthetic_tasks_demo.csv', index=False)