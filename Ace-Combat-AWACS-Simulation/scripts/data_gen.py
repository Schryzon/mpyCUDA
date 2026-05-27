import pandas as pd
import numpy as np
import os
import math

def generate_radar_data(num_records=1000000, output_file='radar_data.csv'):
    print(f"Generating {num_records} synthetic radar records...")
    
    # 1. Define operational Allied bases
    bases = [
        (0.0, 0.0),            # Base Alpha (HQ)
        (-60000.0, 40000.0),   # Base Bravo (FOB)
        (50000.0, -50000.0)    # Base Charlie (Naval)
    ]
    
    # 2. Positions (meters) - distributed over a 200km theater
    angles = np.random.uniform(0, 2 * np.pi, num_records)
    distances_xy = np.random.uniform(10000, 200000, num_records)
    x = distances_xy * np.cos(angles)
    y = distances_xy * np.sin(angles)
    altitude = np.random.uniform(500, 20000, num_records) # 500m to 20km
    
    # 3. IFF Status: 15% Allies (Friendly), 85% Hostile/Unknown Bogeys
    iff_status = (np.random.rand(num_records) < 0.15).astype(int)
    allies_mask = (iff_status == 1)
    enemies_mask = (iff_status == 0)
    
    # 4. Aircraft Types assignment
    # Allies: 50% F-15 (ID 2), 50% F-14 (ID 3)
    # Enemies: 50% MiG-29 (ID 0), 50% Su-27 (ID 1)
    aircraft_type_id = np.zeros(num_records, dtype=int)
    aircraft_type_id[allies_mask] = np.random.choice([2, 3], size=np.sum(allies_mask))
    aircraft_type_id[enemies_mask] = np.random.choice([0, 1], size=np.sum(enemies_mask))
    
    # 5. Speeds based on aircraft class capabilities (m/s)
    velocity = np.zeros(num_records)
    velocity[aircraft_type_id == 0] = np.random.uniform(450, 750, np.sum(aircraft_type_id == 0))  # MiG-29 (Fast/Agile)
    velocity[aircraft_type_id == 1] = np.random.uniform(350, 650, np.sum(aircraft_type_id == 1))  # Su-27 (Heavy/Maneuverable)
    velocity[aircraft_type_id == 2] = np.random.uniform(400, 700, np.sum(aircraft_type_id == 2))  # F-15 Eagle (Allied Interceptor)
    velocity[aircraft_type_id == 3] = np.random.uniform(300, 600, np.sum(aircraft_type_id == 3))  # F-14 Tomcat (Fleet Defense)
    
    # 6. Calculate proximity to the CLOSEST base out of the three
    dists = [np.sqrt((x - bx)**2 + (y - by)**2) for bx, by in bases]
    distances_to_closest = np.minimum.reduce(dists)
    closest_base_idx = np.argmin(np.stack(dists, axis=0), axis=0)
    
    bx_closest = np.choose(closest_base_idx, [bases[0][0], bases[1][0], bases[2][0]])
    by_closest = np.choose(closest_base_idx, [bases[0][1], bases[1][1], bases[2][1]])
    
    # 7. Direction Vectors and Headings relative to nearest base
    angle_to_base_rad = np.arctan2(by_closest - y, bx_closest - x)
    angle_to_base_deg = (np.degrees(angle_to_base_rad) + 360) % 360
    
    # Heading (degrees)
    heading = np.random.uniform(0, 360, num_records)
    
    # Make ~30% of enemies perform an intercept run toward their nearest base
    targeting_closest_base = (np.random.rand(num_records) < 0.3) & enemies_mask
    heading[targeting_closest_base] = np.random.normal(angle_to_base_deg[targeting_closest_base], 5.0)
    heading = heading % 360
    
    # shortest angular distance to closest base
    heading_diff = np.abs(heading - angle_to_base_deg)
    heading_diff = np.minimum(heading_diff, 360 - heading_diff)
    
    # 8. Threat score calculation (0 to 100) - Only calculated for hostiles!
    score = np.zeros(num_records)
    # Proximity Factor (up to 35 pts)
    score[enemies_mask] += np.clip(35 - (distances_to_closest[enemies_mask] / 6000.0), 0, 35)
    # Speed Factor (up to 30 pts)
    score[enemies_mask] += np.clip((velocity[enemies_mask] - 170) / (850 - 170) * 30, 0, 30)
    # Vector Alignment Factor (up to 30 pts)
    score[enemies_mask] += np.clip(30 - (heading_diff[enemies_mask] / 3.0), 0, 30)
    # Altitude Factor (up to 5 pts)
    score[enemies_mask] += np.clip((20000 - altitude[enemies_mask]) / 19500.0 * 5, 0, 5)
    
    # 9. Categorize Threat Labels (0 = Low, 1 = Medium, 2 = High, 3 = Critical)
    # Allies always have threat level 0 (Low)
    threat_label = np.zeros(num_records, dtype=int)
    threat_label[enemies_mask & (score > 35)] = 1
    threat_label[enemies_mask & (score > 55)] = 2
    threat_label[enemies_mask & (score > 70)] = 3
    
    # Inject 1% faulty sensor/IFF noise to keep the ML training challenging
    noise_mask = (np.random.rand(num_records) < 0.01) & enemies_mask
    threat_label[noise_mask] = np.random.randint(0, 4, np.sum(noise_mask))
    
    # Map ID to string names
    type_mapping = {0: 'MiG-29', 1: 'Su-27', 2: 'F-15', 3: 'F-14'}
    aircraft_type = np.array([type_mapping[tid] for tid in aircraft_type_id])
    
    df = pd.DataFrame({
        'bogey_id': np.arange(1, num_records + 1),
        'x': x,
        'y': y,
        'altitude': altitude,
        'velocity': velocity,
        'heading': heading,
        'iff_status': iff_status,
        'aircraft_type': aircraft_type,
        'aircraft_type_id': aircraft_type_id,
        'threat_label': threat_label
    })
    
    df = df.round(2)
    df.to_csv(output_file, index=False)
    print(f"Tactical radar data saved to {output_file} successfully.")

if __name__ == "__main__":
    import sys
    num = 1000000
    if len(sys.argv) > 1:
        num = int(sys.argv[1])
    generate_radar_data(num)
