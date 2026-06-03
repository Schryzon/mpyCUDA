import pandas as pd
import numpy as np
import os
import math

def generate_formation_offsets(formation_type, size, spacing=1200.0):
    """
    Generates 3D offsets (dx, dy, dz) for wingmen in a formation.
    The leader is at index 0 and has offset (0, 0, 0).
    """
    offsets = []
    
    if formation_type == 'v' or formation_type == 'vic':
        for i in range(size):
            if i == 0:
                offsets.append((0.0, 0.0, 0.0))
            else:
                side = 1.0 if (i % 2 == 1) else -1.0
                step = (i + 1) // 2
                offsets.append((side * step * spacing, -step * spacing, np.random.uniform(-100.0, 100.0)))
                
    elif formation_type == 'echelon_right':
        for i in range(size):
            offsets.append((i * spacing, -i * spacing, np.random.uniform(-50.0, 50.0)))
            
    elif formation_type == 'echelon_left':
        for i in range(size):
            offsets.append((-i * spacing, -i * spacing, np.random.uniform(-50.0, 50.0)))
            
    elif formation_type == 'finger_four':
        # Classic military finger-four formation
        finger_offsets = [
            (0.0, 0.0, 0.0),
            (-spacing, -spacing, np.random.uniform(-50.0, 50.0)),
            (2.0 * spacing, -0.5 * spacing, np.random.uniform(-50.0, 50.0)),
            (spacing, -1.5 * spacing, np.random.uniform(-50.0, 50.0))
        ]
        for i in range(size):
            offsets.append(finger_offsets[i % 4])
            
    elif formation_type == 'double_v':
        # Double V (nested V elements)
        for i in range(size):
            if i < 6:
                side = 1.0 if (i % 2 == 1) else -1.0
                step = (i + 1) // 2
                offsets.append((side * step * spacing, -step * spacing, np.random.uniform(-100.0, 100.0)))
            else:
                idx = i - 6
                side = 1.0 if (idx % 2 == 1) else -1.0
                step = (idx + 1) // 2
                offsets.append((side * step * spacing, -step * spacing - 4.0 * spacing, np.random.uniform(-150.0, -50.0)))
                
    else: # Default fallback: simple trail
        for i in range(size):
            offsets.append((0.0, -i * spacing, 0.0))
            
    return offsets

def rotate_offsets(offsets, heading_deg):
    """
    Rotates 3D offsets relative to the leader's flight heading (degrees).
    Heading is measured clockwise from North (+y).
    """
    heading_rad = heading_deg * np.pi / 180.0
    c = np.cos(heading_rad)
    s = np.sin(heading_rad)
    
    rotated = []
    for dx, dy, dz in offsets:
        # Perpendicular vector rotation (North is +y, East is +x)
        rx = dx * c + dy * s
        ry = -dx * s + dy * c
        rotated.append((rx, ry, dz))
        
    return rotated

def generate_radar_data(num_records=1000000, output_file='radar_data.csv'):
    print(f"Generating {num_records} tactical radar records with Ace Combat elite squadrons...")
    
    # 1. Define active military bases
    bases = [
        (0.0, 0.0),            # Base Alpha (HQ)
        (-60000.0, 40000.0),   # Base Bravo (FOB)
        (50000.0, -50000.0)    # Base Charlie (Naval)
    ]
    
    # Pre-allocate output arrays
    x = np.zeros(num_records)
    y = np.zeros(num_records)
    altitude = np.zeros(num_records)
    velocity = np.zeros(num_records)
    heading = np.zeros(num_records)
    iff_status = np.zeros(num_records, dtype=int)
    aircraft_type_id = np.zeros(num_records, dtype=int)
    squadron_name = np.array(['None'] * num_records, dtype=object)
    callsign = np.array(['None'] * num_records, dtype=object)
    
    current_idx = 0
    
    # 2. Inject Elite Hostile squadrons first (these are our active scenario bogeys)
    # We want to place them close to bases heading towards them, ensuring high threat classification.
    elite_bogey_squadrons = [
        {
            'name': 'Grabacr',
            'type_id': 9, # Su-47
            'size': 4,
            'formation': 'finger_four',
            'base_target': 0, # Alpha
            'altitude_range': (8000, 12000),
            'speed_range': (720, 800)
        },
        {
            'name': 'Gelb',
            'type_id': 10, # Su-37
            'size': 2,
            'formation': 'echelon_left',
            'base_target': 1, # Bravo
            'altitude_range': (6000, 8000),
            'speed_range': (650, 720)
        },
        {
            'name': 'Strigon',
            'type_id': 11, # Su-33
            'size': 12,
            'formation': 'double_v',
            'base_target': 2, # Charlie
            'altitude_range': (5000, 9000),
            'speed_range': (550, 650)
        },
        {
            'name': 'Sol',
            'type_id': 12, # Su-30SM
            'size': 5,
            'formation': 'v',
            'base_target': 0, # Alpha
            'altitude_range': (7000, 10000),
            'speed_range': (620, 700)
        },
        {
            'name': 'Yellow',
            'type_id': 10, # Su-37
            'size': 5,
            'formation': 'v',
            'base_target': 2, # Charlie
            'altitude_range': (9000, 13000),
            'speed_range': (680, 760)
        },
        {
            'name': 'Ofnir',
            'type_id': 13, # Su-35
            'size': 4,
            'formation': 'finger_four',
            'base_target': 1, # Bravo
            'altitude_range': (8000, 11000),
            'speed_range': (600, 680)
        },
        {
            'name': 'Raven',
            'type_id': 16, # ADF-11F
            'size': 4,
            'formation': 'finger_four',
            'base_target': 0, # Alpha
            'altitude_range': (10000, 12000),
            'speed_range': (800, 900)
        },
        {
            'name': 'Crimson',
            'type_id': 17, # PW-Mk.I
            'size': 2,
            'formation': 'echelon_right',
            'base_target': 1, # Bravo
            'altitude_range': (9000, 11000),
            'speed_range': (750, 850)
        },
        {
            'name': 'Gleipnir',
            'type_id': 19, # Enemy airship
            'size': 1,
            'formation': 'trail',
            'base_target': 2, # Charlie
            'altitude_range': (9000, 10000),
            'speed_range': (120, 140)
        }
    ]
    
    elite_allied_squadrons = [
        {
            'name': 'Faust',
            'type_id': 18, # Allied airship
            'size': 1,
            'formation': 'trail',
            'base_target': 2, # Charlie
            'altitude_range': (8000, 9000),
            'speed_range': (120, 130)
        },
        {
            'name': 'Falken',
            'type_id': 14, # ADF-01
            'size': 2,
            'formation': 'echelon_right',
            'base_target': 0, # Alpha
            'altitude_range': (8000, 10000),
            'speed_range': (700, 800)
        },
        {
            'name': 'Wyvern',
            'type_id': 15, # X-02S
            'size': 2,
            'formation': 'echelon_left',
            'base_target': 1, # Bravo
            'altitude_range': (8000, 10000),
            'speed_range': (750, 850)
        }
    ]
    
    # We generate multiple flights of these elite squadrons scattered across different sectors
    # to populate the critical defense airspace.
    num_flights = 5
    for flight_num in range(num_flights):
        # Generate bogeys
        for sq in elite_bogey_squadrons:
            sq_size = sq['size']
            
            # Check if there is enough space left in the array
            if current_idx + sq_size > num_records * 0.12: # Limit elite units to 12% of workspace max
                break
                
            # Random starting position for squadron leader, vectoring in on their targeted base
            target_bx, target_by = bases[sq['base_target']]
            
            # Distance: 15km to 60km out
            dist_to_base = np.random.uniform(15000.0, 60000.0)
            angle_from_base = np.random.uniform(0.0, 2.0 * np.pi)
            
            leader_x = target_bx + dist_to_base * np.cos(angle_from_base)
            leader_y = target_by + dist_to_base * np.sin(angle_from_base)
            
            # Aim directly at the targeted base
            angle_to_base_rad = np.arctan2(target_by - leader_y, target_bx - leader_x)
            leader_heading = (90.0 - np.degrees(angle_to_base_rad) + 360.0) % 360.0
            
            leader_alt = np.random.uniform(*sq['altitude_range'])
            leader_speed = np.random.uniform(*sq['speed_range'])
            
            offsets = generate_formation_offsets(sq['formation'], sq_size)
            rotated = rotate_offsets(offsets, leader_heading)
            
            for i in range(sq_size):
                idx = current_idx + i
                rx, ry, rz = rotated[i]
                
                x[idx] = leader_x + rx
                y[idx] = leader_y + ry
                altitude[idx] = np.clip(leader_alt + rz, 100.0, 22000.0)
                velocity[idx] = leader_speed + np.random.normal(0.0, 2.0)
                heading[idx] = (leader_heading + np.random.normal(0.0, 0.5)) % 360.0
                iff_status[idx] = 0 # Enemy Bogey
                aircraft_type_id[idx] = sq['type_id']
                squadron_name[idx] = sq['name']
                
                # Special callsign mapping
                if sq['name'] == 'Yellow' and i == 0:
                    callsign[idx] = 'Yellow 13'
                else:
                    callsign[idx] = f"{sq['name']} {i + 1}"
                    
            current_idx += sq_size

        # Generate elite allies
        for sq in elite_allied_squadrons:
            sq_size = sq['size']
            
            if current_idx + sq_size > num_records * 0.15:
                break
                
            target_bx, target_by = bases[sq['base_target']]
            dist_to_base = np.random.uniform(5000.0, 30000.0)
            angle_from_base = np.random.uniform(0.0, 2.0 * np.pi)
            
            leader_x = target_bx + dist_to_base * np.cos(angle_from_base)
            leader_y = target_by + dist_to_base * np.sin(angle_from_base)
            leader_heading = np.random.uniform(0.0, 360.0)
            
            leader_alt = np.random.uniform(*sq['altitude_range'])
            leader_speed = np.random.uniform(*sq['speed_range'])
            
            offsets = generate_formation_offsets(sq['formation'], sq_size)
            rotated = rotate_offsets(offsets, leader_heading)
            
            for i in range(sq_size):
                idx = current_idx + i
                rx, ry, rz = rotated[i]
                
                x[idx] = leader_x + rx
                y[idx] = leader_y + ry
                altitude[idx] = np.clip(leader_alt + rz, 100.0, 22000.0)
                velocity[idx] = leader_speed + np.random.normal(0.0, 2.0)
                heading[idx] = (leader_heading + np.random.normal(0.0, 0.5)) % 360.0
                iff_status[idx] = 1 # Allied Friendly
                aircraft_type_id[idx] = sq['type_id']
                squadron_name[idx] = sq['name']
                callsign[idx] = f"{sq['name']} {i + 1}"
                
            current_idx += sq_size

    print(f"Generated {current_idx} elite bogey squadron records successfully.")
    
    # 3. Fill the remaining records with randomly scattered background flights
    remaining = num_records - current_idx
    
    # Distributed over a wide 200km theater
    angles = np.random.uniform(0.0, 2.0 * np.pi, remaining)
    distances_xy = np.random.uniform(10000.0, 200000.0, remaining)
    
    x[current_idx:] = distances_xy * np.cos(angles)
    y[current_idx:] = distances_xy * np.sin(angles)
    altitude[current_idx:] = np.random.uniform(500.0, 20000.0, remaining)
    
    # IFF status: 15% allies (Friendly), 85% hostile/unknown bogeys
    remaining_iff = (np.random.rand(remaining) < 0.15).astype(int)
    iff_status[current_idx:] = remaining_iff
    
    allies_mask = (remaining_iff == 1)
    enemies_mask = (remaining_iff == 0)
    
    # Standard and Advanced plane type IDs:
    # Allies: F-15 (2), F-14 (3), ADF-01 (14), X-02S (15), P-1112 Airship (18)
    # Enemies: MiG-29 (0), Su-27 (1), ADF-11F (16), PW-Mk.I (17), GLEIPNIR Airship (19)
    allies_types = [2, 3, 14, 15, 18]
    allies_p = [0.44, 0.44, 0.05, 0.05, 0.02]
    
    enemies_types = [0, 1, 16, 17, 19]
    enemies_p = [0.44, 0.44, 0.05, 0.05, 0.02]
    
    rem_type_ids = np.zeros(remaining, dtype=int)
    rem_type_ids[allies_mask] = np.random.choice(allies_types, size=np.sum(allies_mask), p=allies_p)
    rem_type_ids[enemies_mask] = np.random.choice(enemies_types, size=np.sum(enemies_mask), p=enemies_p)
    aircraft_type_id[current_idx:] = rem_type_ids
    
    # Velocities based on capabilities
    rem_velocities = np.zeros(remaining)
    rem_velocities[rem_type_ids == 0] = np.random.uniform(450.0, 750.0, np.sum(rem_type_ids == 0)) # MiG-29
    rem_velocities[rem_type_ids == 1] = np.random.uniform(350.0, 650.0, np.sum(rem_type_ids == 1)) # Su-27
    rem_velocities[rem_type_ids == 2] = np.random.uniform(400.0, 700.0, np.sum(rem_type_ids == 2)) # F-15
    rem_velocities[rem_type_ids == 3] = np.random.uniform(300.0, 600.0, np.sum(rem_type_ids == 3)) # F-14
    rem_velocities[rem_type_ids == 14] = np.random.uniform(700.0, 850.0, np.sum(rem_type_ids == 14)) # ADF-01 Falken
    rem_velocities[rem_type_ids == 15] = np.random.uniform(750.0, 900.0, np.sum(rem_type_ids == 15)) # X-02S Strike Wyvern
    rem_velocities[rem_type_ids == 16] = np.random.uniform(800.0, 950.0, np.sum(rem_type_ids == 16)) # ADF-11F Raven
    rem_velocities[rem_type_ids == 17] = np.random.uniform(750.0, 900.0, np.sum(rem_type_ids == 17)) # PW-Mk.I
    rem_velocities[rem_type_ids == 18] = np.random.uniform(120.0, 150.0, np.sum(rem_type_ids == 18)) # P-1112 Aigaion
    rem_velocities[rem_type_ids == 19] = np.random.uniform(120.0, 140.0, np.sum(rem_type_ids == 19)) # GLEIPNIR
    velocity[current_idx:] = rem_velocities
    
    # Proximity calculation to assign heading vectors for standard hostiles
    rem_x = x[current_idx:]
    rem_y = y[current_idx:]
    
    dists = [np.sqrt((rem_x - bx)**2 + (rem_y - by)**2) for bx, by in bases]
    closest_base_idx = np.argmin(np.stack(dists, axis=0), axis=0)
    
    bx_closest = np.choose(closest_base_idx, [bases[0][0], bases[1][0], bases[2][0]])
    by_closest = np.choose(closest_base_idx, [bases[0][1], bases[1][1], bases[2][1]])
    
    angle_to_base_rad = np.arctan2(by_closest - rem_y, bx_closest - rem_x)
    angle_to_base_deg = (90.0 - np.degrees(angle_to_base_rad) + 360.0) % 360.0
    
    rem_headings = np.random.uniform(0.0, 360.0, remaining)
    
    # 30% of standard enemies perform direct intercept runs on closest base
    intercept_mask = (np.random.rand(remaining) < 0.3) & enemies_mask
    rem_headings[intercept_mask] = np.random.normal(angle_to_base_deg[intercept_mask], 5.0)
    heading[current_idx:] = rem_headings % 360.0
    
    # 4. Threat Scoring & Classification
    # Proximity to closest base (out of all bases)
    all_dists = [np.sqrt((x - bx)**2 + (y - by)**2) for bx, by in bases]
    distances_to_closest = np.minimum.reduce(all_dists)
    closest_base_idx_global = np.argmin(np.stack(all_dists, axis=0), axis=0)
    
    bx_closest_global = np.choose(closest_base_idx_global, [bases[0][0], bases[1][0], bases[2][0]])
    by_closest_global = np.choose(closest_base_idx_global, [bases[0][1], bases[1][1], bases[2][1]])
    
    angle_to_base_rad_global = np.arctan2(by_closest_global - y, bx_closest_global - x)
    angle_to_base_deg_global = (90.0 - np.degrees(angle_to_base_rad_global) + 360.0) % 360.0
    
    heading_diff = np.abs(heading - angle_to_base_deg_global)
    heading_diff = np.minimum(heading_diff, 360.0 - heading_diff)
    
    enemies_mask_global = (iff_status == 0)
    
    score = np.zeros(num_records)
    # Proximity Factor (up to 35 pts)
    score[enemies_mask_global] += np.clip(35.0 - (distances_to_closest[enemies_mask_global] / 6000.0), 0.0, 35.0)
    # Speed Factor (up to 30 pts)
    score[enemies_mask_global] += np.clip((velocity[enemies_mask_global] - 170.0) / (850.0 - 170.0) * 30.0, 0.0, 30.0)
    # Vector Alignment Factor (up to 30 pts)
    score[enemies_mask_global] += np.clip(30.0 - (heading_diff[enemies_mask_global] / 3.0), 0.0, 30.0)
    # Altitude Factor (up to 5 pts)
    score[enemies_mask_global] += np.clip((20000.0 - altitude[enemies_mask_global]) / 19500.0 * 5.0, 0.0, 5.0)
    
    # Categorize Threat Labels (0 = Low, 1 = Medium, 2 = High, 3 = Critical)
    threat_label = np.zeros(num_records, dtype=int)
    threat_label[enemies_mask_global & (score > 35.0)] = 1
    threat_label[enemies_mask_global & (score > 55.0)] = 2
    threat_label[enemies_mask_global & (score > 70.0)] = 3
    
    # Inject 1% faulty sensor/IFF noise for ML challenge
    noise_mask = (np.random.rand(num_records) < 0.01) & enemies_mask_global
    threat_label[noise_mask] = np.random.randint(0, 4, np.sum(noise_mask))
    
    # Map IDs to Aircraft Names
    type_mapping = {
        0: 'MiG-29', 1: 'Su-27', 2: 'F-15', 3: 'F-14',
        4: 'F-22A', 5: 'F-14D', 6: 'F-15C', 7: 'F-15E', 8: 'F-16C',
        9: 'Su-47', 10: 'Su-37', 11: 'Su-33', 12: 'Su-30SM', 13: 'Su-35',
        14: 'ADF-01', 15: 'X-02S', 16: 'ADF-11F', 17: 'PW-Mk.I',
        18: 'P-1112', 19: 'GLEIPNIR'
    }
    aircraft_type = np.array([type_mapping[tid] for tid in aircraft_type_id])
    
    # Create final DataFrame
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
        'threat_label': threat_label,
        'squadron_name': squadron_name,
        'callsign': callsign
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
