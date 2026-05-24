import pandas as pd
import numpy as np
import os
import math

def generate_radar_data(num_records=1000000, output_file='radar_data.csv'):
    print(f"Generating {num_records} synthetic radar records...")
    
    # Positions (meters)
    # Range: 10km to 200km away
    angles = np.random.uniform(0, 2 * np.pi, num_records)
    distances_xy = np.random.uniform(10000, 200000, num_records)
    x = distances_xy * np.cos(angles)
    y = distances_xy * np.sin(angles)
    altitude = np.random.uniform(500, 20000, num_records) # 500m to 20km
    
    # Velocity (m/s)
    # 170 m/s (Mach 0.5) to 850 m/s (Mach 2.5)
    velocity = np.random.uniform(170, 850, num_records)
    
    # Heading (degrees) 0 to 360
    heading = np.random.uniform(0, 360, num_records)
    
    # Angle to base (0, 0)
    angle_to_base_rad = np.arctan2(-y, -x)
    angle_to_base_deg = (np.degrees(angle_to_base_rad) + 360) % 360
    
    # Make ~30% of targets head towards the base (within +/- 10 degrees)
    targeting_base = np.random.rand(num_records) < 0.3
    heading[targeting_base] = np.random.normal(angle_to_base_deg[targeting_base], 5.0)
    heading = heading % 360
    
    # Calculate Threat Level for ML training (Ground Truth)
    heading_diff = np.abs(heading - angle_to_base_deg)
    heading_diff = np.minimum(heading_diff, 360 - heading_diff) # shortest angular distance
    
    # Threat score calculation (0 to 100)
    score = np.zeros(num_records)
    score += np.clip(35 - (distances_xy / 6000.0), 0, 35)
    score += np.clip((velocity - 170) / (850 - 170) * 30, 0, 30)
    score += np.clip(30 - (heading_diff / 3.0), 0, 30)
    # Altitude Factor: +5 points for flying low (under radar horizon)
    score += np.clip((20000 - altitude) / 19500.0 * 5, 0, 5)
    
    # Labels: 0 = Low, 1 = Medium, 2 = High, 3 = Critical
    threat_label = np.zeros(num_records, dtype=int)
    threat_label[score > 35] = 1
    threat_label[score > 55] = 2
    threat_label[score > 70] = 3
    
    # Inject 1% random noise to simulate faulty IFF/sensors (Realistic ML Task)
    noise_mask = np.random.rand(num_records) < 0.01
    threat_label[noise_mask] = np.random.randint(0, 4, np.sum(noise_mask))
    
    df = pd.DataFrame({
        'bogey_id': np.arange(1, num_records + 1),
        'x': x,
        'y': y,
        'altitude': altitude,
        'velocity': velocity,
        'heading': heading,
        'threat_label': threat_label
    })
    
    # Save slightly rounded values for realism
    df = df.round(2)
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file} successfully.")

if __name__ == "__main__":
    import sys
    num = 1000000
    if len(sys.argv) > 1:
        num = int(sys.argv[1])
    generate_radar_data(num)
