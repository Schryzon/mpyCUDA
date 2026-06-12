import plotly.graph_objects as go
import pandas as pd
import numpy as np
import math
import random

# Railgun Helix generator
def get_railgun_helix(x0, y0, z0, x1, y1, z1):
    hx, hy, hz = [], [], []
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    dist = math.sqrt(dx**2 + dy**2 + dz**2)
    if dist < 1.0:
        return hx, hy, hz
    if abs(dx) > abs(dy):
        nx, ny, nz = -dz, 0.0, dx
    else:
        nx, ny, nz = 0.0, -dz, dy
    n_len = math.sqrt(nx**2 + ny**2 + nz**2)
    nx, ny, nz = nx/n_len, ny/n_len, nz/n_len
    bx = dy*nz - dz*ny
    by = dz*nx - dx*nz
    bz = dx*ny - dy*nx
    b_len = math.sqrt(bx**2 + by**2 + bz**2)
    bx, by, bz = bx/b_len, by/b_len, bz/b_len
    steps = 30
    turns = 5.0
    radius = 500.0
    for s in range(steps + 1):
        u = s / steps
        theta = 2.0 * math.pi * turns * u
        px = x0 + u*dx + radius * (math.cos(theta)*nx + math.sin(theta)*bx)
        py = y0 + u*dy + radius * (math.cos(theta)*ny + math.sin(theta)*by)
        pz = z0 + u*dz + radius * (math.cos(theta)*nz + math.sin(theta)*bz)
        hx.append(px)
        hy.append(py)
        hz.append(pz)
    return hx, hy, hz

def wrap_text(text, width=42):
    lines = []
    for line in text.split("<br>"):
        words = line.split(" ")
        curr_line = []
        curr_len = 0
        for w in words:
            if curr_len + len(w) + 1 > width:
                lines.append(" ".join(curr_line))
                curr_line = [w]
                curr_len = len(w)
            else:
                curr_line.append(w)
                curr_len += len(w) + (1 if curr_len > 0 else 0)
        if curr_line:
            lines.append(" ".join(curr_line))
    return "<br>".join(lines)

# Define 3D Physical Engagement Simulator class
class EngagementSimulator:
    def __init__(self, engagements_df, base_x_arr, base_y_arr, base_z_arr):
        self.engagements = engagements_df.to_dict('records')
        self.base_x = base_x_arr
        self.base_y = base_y_arr
        self.base_z = base_z_arr
        self.base_names = {0: "Alpha (HQ)", 1: "Bravo (FOB)", 2: "Charlie (Naval)"}
        
    def run_simulation(self, max_time=30.0, dt=0.25):
        random.seed(42)
        results = []
        global_chatter = []
        
        for idx, eng in enumerate(self.engagements):
            # Target initial state
            tx = eng['x']
            ty = eng['y']
            tz = eng['altitude']
            tv = eng['velocity']
            th = eng['heading']
            t_type = int(eng['aircraft_type_id'])
            t_name = eng['aircraft_type']
            t_callsign = eng['callsign'] if eng['squadron_name'] != 'None' else t_name
            t_iff = int(eng['iff_status'])
            
            # HP assignment
            is_airship = (t_type >= 18 and t_type <= 26)
            hp = 5 if is_airship else 1
            max_hp = hp
            
            # Target heading rad
            th_rad = th * math.pi / 180.0
            tvx = tv * math.sin(th_rad)
            tvy = tv * math.cos(th_rad)
            tvz = 0.0
            
            # Launch base
            b_idx = int(eng['launch_base_idx'])
            bx = self.base_x[b_idx] if b_idx >= 0 else eng['bx_closest']
            by = self.base_y[b_idx] if b_idx >= 0 else eng['by_closest']
            bz = self.base_z[b_idx] if b_idx >= 0 else 0.0
            b_name = self.base_names.get(b_idx, "Unknown")
            
            # Interceptor / Shooter Scramble
            i_weapon = "missile"
            i_speed = 700.0 # m/s
            
            # Symmetrical Role Logic
            if t_iff == 0:
                # Enemy target -> Allied interceptor
                i_name = "Allied Interceptor"
                if t_type == 9: i_name, i_speed = "Mobius F-22A", 850.0
                elif t_type == 10: i_name, i_speed = "Galm F-15C", 750.0
                elif t_type == 11: i_name, i_speed = "Garuda F-15E", 720.0
                elif t_type == 12: i_name, i_speed = "Wardog F-14D", 680.0
                elif t_type == 13: i_name, i_speed = "Crow F-16C", 650.0
                elif t_type in [14, 16, 25]: i_name, i_weapon, i_speed = "FALKEN (TLS)", "laser", 900.0
                elif t_type in [15, 17, 19, 20, 21, 22, 23]: i_name, i_weapon, i_speed = "Stonehenge EML", "railgun", 850.0
                elif t_type == 26: i_name, i_weapon, i_speed = "SOLG Space Strike", "laser", 1200.0
            else:
                # Allied target -> Enemy interceptor
                i_name = "Hostile Interceptor"
                if t_type in [14, 16]: i_name, i_weapon, i_speed = "RAVEN (TLS)", "laser", 900.0
                elif t_type in [15, 17, 18, 24, 25, 26]: i_name, i_weapon, i_speed = "PW-Mk.I (EML)", "railgun", 850.0
                else: i_name, i_speed = "Hostile MiG-29", 680.0
                
            # Interceptor starts either at base or is already airborne (diverted patrol)
            starts_airborne = (random.random() < 0.45)
            if starts_airborne and t_iff == 0:
                ix = bx + 0.4 * (tx - bx)
                iy = by + 0.4 * (ty - by)
                iz = 6000.0 + random.uniform(-1000.0, 2000.0)
            else:
                ix, iy, iz = bx, by, bz
            iv = i_speed
            
            history = []
            missiles = [] # active missiles
            decoys = [] # active decoys
            
            # Weapon status
            chaff_flares_left = 0 if is_airship else 3
            cooldown = 0.0
            target_destroyed = False
            death_time = 9999.0
            target_status = 'active'
            interceptor_status = 'active'
            unit_withdrawn = False
            laser_exposure_time = 0.0
            
            # Initial logs
            role_symbol = "Bandit" if t_iff == 0 else "Allied Patrol"
            if starts_airborne and t_iff == 0:
                global_chatter.append((0.0, f"AWACS: Airborne patrol {i_name} diverted to vector on {role_symbol} {t_callsign} ({t_name})."))
            else:
                global_chatter.append((0.0, f"AWACS: Scramble order issued from Base {b_name}! {i_name} vectoring on {role_symbol} {t_callsign} ({t_name})."))
            if is_airship:
                global_chatter.append((0.0, f"AWACS: Alert! Command Cruiser {t_callsign} is airborne and active!"))
                
            t_sim = 0.0
            while t_sim < max_time and t_sim <= death_time + 1.5 and not unit_withdrawn:
                # 1. Update Target (Evasions/Maneuvers/Withdrawals)
                if not target_destroyed:
                    if is_airship and hp <= 2:
                        if target_status != 'withdrawing':
                            target_status = 'withdrawing'
                            global_chatter.append((t_sim, f"{t_callsign}: Crucial damage. Withdrawing from combat!"))
                    elif not is_airship and chaff_flares_left == 0 and t_sim > 18.0:
                        if target_status != 'withdrawing':
                            target_status = 'withdrawing'
                            global_chatter.append((t_sim, f"{t_callsign}: Out of decoys! Withdrawing and returning to base!"))
                    
                    incoming_missile = False
                    for m in missiles:
                        if m['target_locked'] == 'target':
                            incoming_missile = True
                            break
                            
                    tvz = 0.0
                    if target_status == 'withdrawing':
                        th_rad = math.atan2(tx - bx, ty - by)
                        tv = eng['velocity'] * 1.3
                        tvz = -30.0
                        dist_b = math.sqrt((tx - bx)**2 + (ty - by)**2)
                        if dist_b > 110000.0:
                            unit_withdrawn = True
                            global_chatter.append((t_sim, f"AWACS: {t_callsign} has retreated from tactical radar range."))
                    elif incoming_missile and not is_airship:
                        th_rad += math.sin(t_sim * 2.5) * 0.18 + 0.06
                        tvz = math.cos(t_sim * 2.0) * 80.0
                        
                        if chaff_flares_left > 0 and cooldown <= 0:
                            decoy_type = 'chaff' if t_type in [14, 15, 16, 17] and random.random() > 0.5 else 'flare'
                            decoys.append({
                                'x': tx, 'y': ty, 'z': tz,
                                'vx': tvx * 0.3 + random.uniform(-40, 40),
                                'vy': tvy * 0.3 + random.uniform(-40, 40),
                                'vz': tvz - 30.0,
                                'type': decoy_type,
                                'life': 3.5
                            })
                            chaff_flares_left -= 1
                            cooldown = 4.0
                            global_chatter.append((t_sim, f"{t_callsign}: Locked! Launching {decoy_type.upper()}! Turning hard!"))
                    else:
                        if is_airship:
                            th_rad += 0.005
                        else:
                            th_rad += 0.002
                            
                    tvx = tv * math.sin(th_rad)
                    tvy = tv * math.cos(th_rad)
                    tx += tvx * dt
                    ty += tvy * dt
                    tz = np.clip(tz + tvz * dt, 200.0, 20000.0)
                    
                next_decoys = []
                for d in decoys:
                    d['x'] += d['vx'] * dt
                    d['y'] += d['vy'] * dt
                    d['z'] = max(10.0, d['z'] + d['vz'] * dt)
                    d['vx'] *= 0.82
                    d['vy'] *= 0.82
                    d['vz'] = d['vz'] * 0.88 - 9.8 * dt
                    d['life'] -= dt
                    if d['life'] > 0:
                        next_decoys.append(d)
                decoys = next_decoys
                
                dist_ti = math.sqrt((tx - ix)**2 + (ty - iy)**2 + (tz - iz)**2)
                dx = tx - ix
                dy = ty - iy
                dz = tz - iz
                if dist_ti > 0 and (i_weapon in ['laser', 'railgun'] or not missiles) and not target_destroyed and target_status != 'withdrawing':
                    ix += (dx / dist_ti) * iv * dt
                    iy += (dy / dist_ti) * iv * dt
                    iz += (dz / dist_ti) * iv * dt
                else:
                    dist_to_base = math.sqrt((ix - bx)**2 + (iy - by)**2 + (iz - bz)**2)
                    if dist_to_base > 4000.0:
                        ix += ((bx - ix) / dist_to_base) * iv * dt
                        iy += ((by - iy) / dist_to_base) * iv * dt
                        iz += ((bz - iz) / dist_to_base) * iv * dt
                        if t_sim % 5.0 < 0.1 and not target_destroyed and interceptor_status == 'active':
                            interceptor_status = 'withdrawing'
                            global_chatter.append((t_sim, f"{i_name}: Target retreated/splashed. Withdrawing to base."))
                    
                lasers_active = []
                railguns_active = []
                
                if not target_destroyed and target_status == 'active':
                    if i_weapon == 'laser' and dist_ti < 45000.0:
                        lasers_active.append({
                            'x0': ix, 'y0': iy, 'z0': iz,
                            'x1': tx, 'y1': ty, 'z1': tz
                        })
                        laser_exposure_time += dt
                        if t_sim % 1.5 < 0.2:
                            global_chatter.append((t_sim, f"{i_name}: TLS beam locked. Burning target..."))
                        if laser_exposure_time >= 1.5:
                            target_destroyed = True
                            death_time = t_sim
                            global_chatter.append((t_sim, f"{i_name}: TLS beam sliced wing! Splash one {t_callsign}!"))
                            
                    elif i_weapon == 'railgun' and dist_ti < 35000.0:
                        if t_sim % 5.0 < 0.1:
                            railguns_active.append({
                                'x0': ix, 'y0': iy, 'z0': iz,
                                'x1': tx, 'y1': ty, 'z1': tz
                            })
                            global_chatter.append((t_sim, f"{i_name}: Stonehenge EML active. Firing railgun slug!"))
                            if random.random() < 0.85:
                                hp -= 3
                                if hp <= 0:
                                    target_destroyed = True
                                    death_time = t_sim
                                    global_chatter.append((t_sim, f"AWACS: Splash! Target {t_callsign} destroyed!"))
                                else:
                                    global_chatter.append((t_sim, f"AWACS: EML direct hit! {t_callsign} has heavy armor damage!"))
                            else:
                                global_chatter.append((t_sim, f"{i_name}: Target evaded EML projectile!"))
                                
                    elif i_weapon == 'missile' and dist_ti < 20000.0 and not missiles:
                        missiles.append({
                            'x': ix, 'y': iy, 'z': iz,
                            'vx': (dx / dist_ti) * 1200.0,
                            'vy': (dy / dist_ti) * 1200.0,
                            'vz': (dz / dist_ti) * 1200.0,
                            'target_locked': 'target',
                            'life': 14.0,
                            'path_x': [ix], 'path_y': [iy], 'path_z': [iz]
                        })
                        global_chatter.append((t_sim, f"{i_name}: Fox 3! Missile tracking {t_callsign}!"))
                        
                    if is_airship and dist_ti < 25000.0 and t_sim % 6.0 < 0.2:
                        global_chatter.append((t_sim, f"{t_callsign}: Activating defense grid! Firing SAM battery!"))
                        global_chatter.append((t_sim, f"{i_name}: Threat alert! Missile on tail!"))
                        
                next_missiles = []
                for m in missiles:
                    if m['target_locked'] == 'target':
                        for d_idx, d in enumerate(decoys):
                            dist_md = math.sqrt((m['x'] - d['x'])**2 + (m['y'] - d['y'])**2 + (m['z'] - d['z'])**2)
                            if dist_md < 5000.0:
                                if random.random() < (0.65 if d['type'] == 'flare' else 0.45):
                                    m['target_locked'] = d_idx
                                    global_chatter.append((t_sim, f"AWACS: Target deployed chaff/flares. Lock transferred!"))
                                    break
                                    
                    if m['target_locked'] == 'target':
                        mx_tar, my_tar, mz_tar = tx, ty, tz
                    else:
                        d_idx = m['target_locked']
                        if d_idx < len(decoys):
                            mx_tar, my_tar, mz_tar = decoys[d_idx]['x'], decoys[d_idx]['y'], decoys[d_idx]['z']
                        else:
                            mx_tar, my_tar, mz_tar = m['x'] + m['vx'], m['y'] + m['vy'], m['z'] + m['vz']
                            
                    mdx = mx_tar - m['x']
                    mdy = my_tar - m['y']
                    mdz = mz_tar - m['z']
                    dist_mt = math.sqrt(mdx**2 + mdy**2 + mdz**2)
                    
                    if dist_mt > 0:
                        des_vx = (mdx / dist_mt) * 1200.0
                        des_vy = (mdy / dist_mt) * 1200.0
                        des_vz = (mdz / dist_mt) * 1200.0
                        alpha = 0.28
                        m['vx'] = m['vx'] * (1.0 - alpha) + des_vx * alpha
                        m['vy'] = m['vy'] * (1.0 - alpha) + des_vy * alpha
                        m['vz'] = m['vz'] * (1.0 - alpha) + des_vz * alpha
                        
                        m_speed = math.sqrt(m['vx']**2 + m['vy']**2 + m['vz']**2)
                        m['vx'] = (m['vx'] / m_speed) * 1200.0
                        m['vy'] = (m['vy'] / m_speed) * 1200.0
                        m['vz'] = (m['vz'] / m_speed) * 1200.0
                        
                    m['x'] += m['vx'] * dt
                    m['y'] += m['vy'] * dt
                    m['z'] += m['vz'] * dt
                    m['life'] -= dt
                    
                    m['path_x'].append(m['x'])
                    m['path_y'].append(m['y'])
                    m['path_z'].append(m['z'])
                    
                    hit_dist = math.sqrt((m['x'] - tx)**2 + (m['y'] - ty)**2 + (m['z'] - tz)**2)
                    if hit_dist < 450.0 and not target_destroyed:
                        hp -= 1
                        global_chatter.append((t_sim, f"{i_name}: Direct hit on {t_callsign}!"))
                        if hp <= 0:
                            target_destroyed = True
                            death_time = t_sim
                            global_chatter.append((t_sim, f"AWACS: Splash! Target {t_callsign} destroyed!"))
                        else:
                            global_chatter.append((t_sim, f"AWACS: {t_callsign} remains operational. HP: {hp}/{max_hp}."))
                        continue
                        
                    if m['target_locked'] != 'target':
                        d_idx = m['target_locked']
                        if d_idx < len(decoys):
                            decoy_hit_dist = math.sqrt((m['x'] - decoys[d_idx]['x'])**2 + (m['y'] - decoys[d_idx]['y'])**2 + (m['z'] - decoys[d_idx]['z'])**2)
                            if decoy_hit_dist < 400.0:
                                global_chatter.append((t_sim, f"{i_name}: Missile detonated on flare decoy."))
                                continue
                                
                    if m['life'] > 0:
                        next_missiles.append(m)
                missiles = next_missiles
                
                history.append({
                    'time': t_sim,
                    'tx': tx, 'ty': ty, 'tz': tz,
                    'ix': ix, 'iy': iy, 'iz': iz,
                    'missiles': [{'x': m['x'], 'y': m['y'], 'z': m['z'], 'px': m['path_x'].copy(), 'py': m['path_y'].copy(), 'pz': m['path_z'].copy()} for m in missiles],
                    'flares': [(d['x'], d['y'], d['z']) for d in decoys if d['type'] == 'flare'],
                    'chaff': [(d['x'], d['y'], d['z']) for d in decoys if d['type'] == 'chaff'],
                    'lasers': list(lasers_active),
                    'railguns': list(railguns_active),
                    'hp': hp,
                    'destroyed': target_destroyed,
                    'status': target_status
                })
                
                cooldown -= dt
                t_sim += dt
                
            if not target_destroyed and target_status != 'withdrawing':
                global_chatter.append((t_sim, f"AWACS: Bandit {t_callsign} has bypassed defense borders! Scramble fail."))
                
            results.append({
                'id': idx,
                'callsign': t_callsign,
                'type': t_name,
                'aircraft_type_id': t_type,
                'iff': t_iff,
                'history': history
            })
            
        cap_results = []
        cap_formations = {
            0: [(0.0, 0.0, 0.0), (-1200.0, -1200.0, 100.0), (1200.0, -1200.0, -100.0), (-2400.0, -2400.0, 200.0)],
            1: [(0.0, 0.0, 0.0), (-1200.0, -1200.0, 50.0), (2400.0, -600.0, -50.0), (1200.0, -1800.0, 100.0)],
            2: [(0.0, 0.0, 0.0), (-1200.0, -1200.0, -50.0), (2400.0, -600.0, 50.0), (1200.0, -1800.0, -100.0)]
        }
        cap_names = {0: "Alpha", 1: "Bravo", 2: "Charlie"}
        cap_fighters = []
        
        for b_idx in range(len(self.base_x)):
            base_name_prefix = cap_names.get(b_idx, "Unknown")
            for form_idx in range(len(cap_formations[b_idx])):
                cap_fighters.append({
                    'id': f"CAP-{b_idx}-{form_idx}",
                    'callsign': f"CAP {base_name_prefix} {form_idx+1}",
                    'base_idx': b_idx,
                    'form_idx': form_idx,
                    'status': 'patrolling',
                    'hp': 1,
                    'withdraw_time': random.uniform(18.0, 26.0),
                    'withdraw_start': None,
                    'withdraw_vx': 0.0,
                    'withdraw_vy': 0.0,
                    'death_time': 9999.0,
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'history': []
                })
                
        t_sim = 0.0
        while t_sim <= max_time:
            for c in cap_fighters:
                if c['status'] == 'patrolling':
                    omega = 0.05
                    rad = 20000.0
                    bx, by, bz = self.base_x[c['base_idx']], self.base_y[c['base_idx']], self.base_z[c['base_idx']]
                    lead_x = bx + rad * math.cos(omega * t_sim + c['base_idx'] * math.pi/3.0)
                    lead_y = by + rad * math.sin(omega * t_sim + c['base_idx'] * math.pi/3.0)
                    lead_z = 10000.0 + math.sin(omega * t_sim) * 1200.0
                    
                    h_rad = omega * t_sim + c['base_idx'] * math.pi/3.0 + math.pi/2.0
                    cos_val = math.cos(h_rad)
                    sin_val = math.sin(h_rad)
                    dx, dy, dz = cap_formations[c['base_idx']][c['form_idx']]
                    rx = dx * cos_val + dy * sin_val
                    ry = -dx * sin_val + dy * cos_val
                    cx = lead_x + rx
                    cy = lead_y + ry
                    cz = lead_z + dz
                    c['x'], c['y'], c['z'] = cx, cy, cz
                    
                    for eng in results:
                        if eng['iff'] == 0:
                            hist_t = [h for h in eng['history'] if abs(h['time'] - t_sim) < 0.1]
                            if hist_t:
                                entry = hist_t[0]
                                if not entry['destroyed'] and entry['status'] != 'withdrawing':
                                    bx_dist = math.sqrt((cx - entry['tx'])**2 + (cy - entry['ty'])**2 + (cz - entry['tz'])**2)
                                    if bx_dist < 22000.0 and random.random() < 0.02:
                                        c['status'] = 'destroyed'
                                        c['death_time'] = t_sim
                                        global_chatter.append((t_sim, f"{eng['callsign']}: Locked on allied patrol! Fox 2!"))
                                        global_chatter.append((t_sim + 1.25, f"{c['callsign']}: Mayday! I'm hit! Ejecting!"))
                                        global_chatter.append((t_sim + 1.5, f"AWACS: Lost signal from {c['callsign']}. Splash one ally!"))
                                        break
                                        
                    if c['status'] == 'patrolling' and t_sim > c['withdraw_time']:
                        c['status'] = 'withdrawing'
                        c['withdraw_start'] = t_sim
                        bx, by = self.base_x[c['base_idx']], self.base_y[c['base_idx']]
                        angle = math.atan2(cx - bx, cy - by)
                        c['withdraw_vx'] = 550.0 * math.sin(angle)
                        c['withdraw_vy'] = 550.0 * math.cos(angle)
                        global_chatter.append((t_sim, f"{c['callsign']}: Bingo fuel. Withdrawing from patrol orbit."))
                        
                elif c['status'] == 'withdrawing':
                    c['x'] += c['withdraw_vx'] * dt
                    c['y'] += c['withdraw_vy'] * dt
                    c['z'] = max(100.0, c['z'] - 80.0 * dt)
                    
                    bx, by = self.base_x[c['base_idx']], self.base_y[c['base_idx']]
                    dist = math.sqrt((c['x'] - bx)**2 + (c['y'] - by)**2)
                    if dist > 105000.0:
                        c['status'] = 'withdrawn'
                        
                elif c['status'] == 'destroyed':
                    pass
                    
                c['history'].append({
                    'time': t_sim,
                    'tx': c['x'], 'ty': c['y'], 'tz': c['z'],
                    'status': c['status'],
                    'destroyed': (c['status'] == 'destroyed')
                })
            t_sim += dt
            
        return results, cap_fighters, global_chatter

def show_tactical_map(top_threats, base_x_arr, base_y_arr, base_z_arr, predictions=None):
    fig = go.Figure(data=[
        go.Scatter3d(x=[base_x_arr[0]], y=[base_y_arr[0]], z=[base_z_arr[0]], mode='markers', marker=dict(size=12, color='lime', symbol='square'), name='Base Alpha (HQ)'),
        go.Scatter3d(x=[base_x_arr[1]], y=[base_y_arr[1]], z=[base_z_arr[1]], mode='markers', marker=dict(size=12, color='cyan', symbol='square'), name='Base Bravo (FOB)'),
        go.Scatter3d(x=[base_x_arr[2]], y=[base_y_arr[2]], z=[base_z_arr[2]], mode='markers', marker=dict(size=12, color='magenta', symbol='square'), name='Base Charlie (Naval)')
    ])

    theta = np.linspace(0, 2.*np.pi, 30)
    phi = np.linspace(0, np.pi/2, 15)
    theta, phi = np.meshgrid(theta, phi)
    r = 100000

    x_d = (r * np.sin(phi) * np.cos(theta)).flatten()
    y_d = (r * np.sin(phi) * np.sin(theta)).flatten()
    z_d = (r * np.cos(phi)).flatten()

    fig.add_trace(go.Mesh3d(x=x_d + base_x_arr[0], y=y_d + base_y_arr[0], z=z_d + base_z_arr[0], opacity=0.03, color='lime', alphahull=0, name='Dome Alpha (HQ)', showlegend=True))
    fig.add_trace(go.Mesh3d(x=x_d + base_x_arr[1], y=y_d + base_y_arr[1], z=z_d + base_z_arr[1], opacity=0.03, color='cyan', alphahull=0, name='Dome Bravo (FOB)', showlegend=True))
    fig.add_trace(go.Mesh3d(x=x_d + base_x_arr[2], y=y_d + base_y_arr[2], z=z_d + base_z_arr[2], opacity=0.03, color='magenta', alphahull=0, name='Dome Charlie (Naval)', showlegend=True))

    fig.add_trace(go.Scatter3d(
        x=top_threats['x'], y=top_threats['y'], z=top_threats['altitude'],
        mode='markers', marker=dict(size=2, color='rgba(255, 0, 0, 0.15)'),
        name='Threat Radar Blips'
    ))

    elite_engagements = top_threats[(top_threats['tti'] > 0) & (top_threats['squadron_name'] != 'None')].head(15)
    if len(elite_engagements) < 12:
        standard_engagements = top_threats[(top_threats['tti'] > 0) & (top_threats['squadron_name'] == 'None')].head(12 - len(elite_engagements))
        engagements = pd.concat([elite_engagements, standard_engagements]).copy()
    else:
        engagements = elite_engagements.copy()

    if predictions is not None:
        try:
            from pyspark.sql.functions import col
            allied_elites = predictions.filter((col("iff_status") == 1) & (col("squadron_name") != 'None')).limit(5).toPandas()
            if not allied_elites.empty:
                allied_elites['tti'] = 25.0
                allied_elites['launch_base_idx'] = 0
                allied_elites['int_x'] = allied_elites['x'] + 10000.0
                allied_elites['int_y'] = allied_elites['y'] + 10000.0
                allied_elites['int_z'] = allied_elites['altitude']
                engagements = pd.concat([engagements, allied_elites]).copy()
        except Exception as e:
            print("Could not query allied elites:", e)

    engagements = engagements.reset_index(drop=True)

    simulator = EngagementSimulator(engagements, base_x_arr, base_y_arr, base_z_arr)
    eng_results, cap_results, chatter_log = simulator.run_simulation(max_time=30.0, dt=0.25)
    chatter_log.sort(key=lambda x: x[0])

    hostile_x, hostile_y, hostile_z, hostile_txt, hostile_sym, hostile_size = [], [], [], [], [], []
    allied_x, allied_y, allied_z, allied_txt, allied_sym, allied_size = [], [], [], [], [], []

    for eng in eng_results:
        h0 = eng['history'][0]
        is_airship = (eng['aircraft_type_id'] >= 18 and eng['aircraft_type_id'] <= 26)
        sym = 'square' if is_airship else 'diamond'
        sz = 35 if is_airship else 8
        
        if eng['iff'] == 0:
            hostile_x.append(h0['tx'])
            hostile_y.append(h0['ty'])
            hostile_z.append(h0['tz'])
            hostile_txt.append(eng['callsign'])
            hostile_sym.append(sym)
            hostile_size.append(sz)
            
            allied_x.append(h0['ix'])
            allied_y.append(h0['iy'])
            allied_z.append(h0['iz'])
            allied_txt.append("Allied Fighter")
            allied_sym.append('diamond')
            allied_size.append(8)
        else:
            allied_x.append(h0['tx'])
            allied_y.append(h0['ty'])
            allied_z.append(h0['tz'])
            allied_txt.append(eng['callsign'])
            allied_sym.append(sym)
            allied_size.append(sz)
            
            hostile_x.append(h0['ix'])
            hostile_y.append(h0['iy'])
            hostile_z.append(h0['iz'])
            hostile_txt.append("Hostile Scramble")
            hostile_sym.append('diamond')
            hostile_size.append(8)

    fig.add_trace(go.Scatter3d(
        x=hostile_x, y=hostile_y, z=hostile_z,
        mode='markers+text', marker=dict(size=hostile_size, color='red', symbol=hostile_sym),
        text=hostile_txt, textposition='top center',
        name='Hostile Fleet'
    ))

    fig.add_trace(go.Scatter3d(
        x=allied_x, y=allied_y, z=allied_z,
        mode='markers+text', marker=dict(size=allied_size, color='cyan', symbol=allied_sym),
        text=allied_txt, textposition='bottom center',
        name='Allied Scrambles'
    ))

    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers', marker=dict(size=1, color='orange', symbol='circle'),
        name='Air Explosions'
    ))

    patrol_x, patrol_y, patrol_z = [], [], []
    for cap in cap_results:
        h0 = cap['history'][0]
        patrol_x.append(h0['tx'])
        patrol_y.append(h0['ty'])
        patrol_z.append(h0['tz'])

    fig.add_trace(go.Scatter3d(
        x=patrol_x, y=patrol_y, z=patrol_z,
        mode='markers+text', marker=dict(size=7, color='rgba(0, 255, 200, 0.7)', symbol='diamond'),
        text=[c['callsign'] for c in cap_results],
        textposition='top center',
        name='CAP Patrols'
    ))

    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='lines', line=dict(color='rgba(0, 255, 255, 0.55)', width=2), name='Allied Trails'))
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='lines', line=dict(color='rgba(255, 80, 80, 0.55)', width=2), name='Enemy Trails'))
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='lines', line=dict(color='rgba(255, 255, 0, 0.8)', width=1.5), name='Missile Homing Lines'))
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', marker=dict(size=4, color='rgba(255, 215, 0, 0.85)', symbol='circle'), name='Flares (Heat Decoy)'))
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', marker=dict(size=4, color='rgba(200, 200, 255, 0.85)', symbol='circle'), name='Chaff (Radar Decoy)'))
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='lines', line=dict(color='rgba(255, 0, 100, 0.85)', width=3), name='TLS Laser Beams'))
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='lines', line=dict(color='rgba(150, 50, 255, 0.85)', width=1.5, dash='dash'), name='EML Railgun Electric Trails'))

    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='lines',
        line=dict(color='rgba(255, 68, 68, 0.85)', width=2.0, dash='dash'),
        name='Lock-On Target Alert'
    ))

    num_frames = 65
    times = np.linspace(0.0, 30.0, num_frames)

    xs_all = list(base_x_arr)
    ys_all = [base_y_arr[0], base_y_arr[1], base_y_arr[2]]
    zs_all = [0.0, 0.0, 0.0]
    for eng in eng_results:
        for h in eng['history']:
            xs_all.extend([h['tx'], h['ix']])
            ys_all.extend([h['ty'], h['iy']])
            zs_all.extend([h['tz'], h['iz']])
    for cap in cap_results:
        for h in cap['history']:
            if h['status'] != 'withdrawn':
                xs_all.append(h['tx'])
                ys_all.append(h['ty'])
                zs_all.append(h['tz'])

    pad_val = 15000.0
    x_min, x_max = min(xs_all) - pad_val, max(xs_all) + pad_val
    y_min, y_max = min(ys_all) - pad_val, max(ys_all) + pad_val
    z_min, z_max = 0.0, max(22000.0, max(zs_all) + 2000.0)

    frames = []
    for k, t in enumerate(times):
        h_x, h_y, h_z, h_txt, h_sym, h_sz = [], [], [], [], [], []
        a_x, a_y, a_z, a_txt, a_sym, a_sz = [], [], [], [], [], []
        exp_x, exp_y, exp_z, exp_sz = [], [], [], []
        lock_x, lock_y, lock_z = [], [], []
        
        allied_trail_x, allied_trail_y, allied_trail_z = [], [], []
        enemy_trail_x, enemy_trail_y, enemy_trail_z = [], [], []
        missile_trail_x, missile_trail_y, missile_trail_z = [], [], []
        flare_x, flare_y, flare_z = [], [], []
        chaff_x, chaff_y, chaff_z = [], [], []
        laser_x, laser_y, laser_z = [], [], []
        railgun_x, railgun_y, railgun_z = [], [], []
        
        for eng in eng_results:
            hist = eng['history']
            entry = min(hist, key=lambda x: abs(x['time'] - t))
            is_airship = (eng['aircraft_type_id'] >= 18 and eng['aircraft_type_id'] <= 26)
            sym = 'square' if is_airship else 'diamond'
            sz = 35 if is_airship else 8
            
            e_path = [ (h['tx'], h['ty'], h['tz']) for h in hist if h['time'] <= entry['time'] ]
            i_path = [ (h['ix'], h['iy'], h['iz']) for h in hist if h['time'] <= entry['time'] ]
            
            if eng['iff'] == 0:
                for px, py, pz in e_path:
                    enemy_trail_x.append(px)
                    enemy_trail_y.append(py)
                    enemy_trail_z.append(pz)
                enemy_trail_x.append(None)
                enemy_trail_y.append(None)
                enemy_trail_z.append(None)
                
                for px, py, pz in i_path:
                    allied_trail_x.append(px)
                    allied_trail_y.append(py)
                    allied_trail_z.append(pz)
                allied_trail_x.append(None)
                allied_trail_y.append(None)
                allied_trail_z.append(None)
            else:
                for px, py, pz in e_path:
                    allied_trail_x.append(px)
                    allied_trail_y.append(py)
                    allied_trail_z.append(pz)
                allied_trail_x.append(None)
                allied_trail_y.append(None)
                allied_trail_z.append(None)
                
                for px, py, pz in i_path:
                    enemy_trail_x.append(px)
                    enemy_trail_y.append(py)
                    enemy_trail_z.append(pz)
                enemy_trail_x.append(None)
                enemy_trail_y.append(None)
                enemy_trail_z.append(None)
            
            if not entry['destroyed'] or entry['time'] >= t - 1.5:
                px_t, py_t, pz_t = entry['tx'], entry['ty'], entry['tz']
                ix_t, iy_t, iz_t = entry['ix'], entry['iy'], entry['iz']
                
                is_locked_on = False
                for m in entry['missiles']:
                    is_locked_on = True
                
                if not is_locked_on:
                    dist_lock = math.sqrt((entry['tx'] - entry['ix'])**2 + (entry['ty'] - entry['iy'])**2 + (entry['tz'] - entry['iz'])**2)
                    if dist_lock < 25000.0 and not entry['destroyed'] and entry['status'] != 'withdrawing':
                        is_locked_on = True
                
                label_txt = f"⚠️ [LOCK] {eng['callsign']}" if is_locked_on else eng['callsign']
                
                if eng['iff'] == 0:
                    h_x.append(px_t)
                    h_y.append(py_t)
                    h_z.append(pz_t)
                    h_txt.append(label_txt)
                    h_sym.append(sym)
                    h_sz.append(sz)
                    
                    a_x.append(ix_t)
                    a_y.append(iy_t)
                    a_z.append(iz_t)
                    a_txt.append("Allied Fighter")
                    a_sym.append('diamond')
                    a_sz.append(8)
                else:
                    a_x.append(px_t)
                    a_y.append(py_t)
                    a_z.append(pz_t)
                    a_txt.append(label_txt)
                    a_sym.append(sym)
                    a_sz.append(sz)
                    
                    h_x.append(ix_t)
                    h_y.append(iy_t)
                    h_z.append(iz_t)
                    h_txt.append("Hostile Scramble")
                    h_sym.append('diamond')
                    h_sz.append(8)
                    
                if entry['destroyed'] and entry['time'] <= t:
                    exp_x.append(px_t)
                    exp_y.append(py_t)
                    exp_z.append(pz_t)
                    exp_sz.append(int(12 + (t - entry['time']) * 14))
                    
            missile_locked = False
            for m in entry['missiles']:
                for mx_p, my_p, mz_p in zip(m['px'], m['py'], m['pz']):
                    missile_trail_x.append(mx_p)
                    missile_trail_y.append(my_p)
                    missile_trail_z.append(mz_p)
                missile_trail_x.append(None)
                missile_trail_y.append(None)
                missile_trail_z.append(None)
                
                lock_x.extend([m['x'], entry['tx'], None])
                lock_y.extend([m['y'], entry['ty'], None])
                lock_z.extend([m['z'], entry['tz'], None])
                missile_locked = True
                
            if not missile_locked and not entry['destroyed'] and entry['status'] != 'withdrawing':
                dist_lock = math.sqrt((entry['tx'] - entry['ix'])**2 + (entry['ty'] - entry['iy'])**2 + (entry['tz'] - entry['iz'])**2)
                if dist_lock < 25000.0:
                    lock_x.extend([entry['ix'], entry['tx'], None])
                    lock_y.extend([entry['iy'], entry['ty'], None])
                    lock_z.extend([entry['iz'], entry['tz'], None])
                
            for fx_p, fy_p, fz_p in entry['flares']:
                flare_x.append(fx_p)
                flare_y.append(fy_p)
                flare_z.append(fz_p)
                
            for cx_p, cy_p, cz_p in entry['chaff']:
                chaff_x.append(cx_p)
                chaff_y.append(cy_p)
                chaff_z.append(cz_p)
                
            for l in entry['lasers']:
                laser_x.extend([l['x0'], l['x1'], None])
                laser_y.extend([l['y0'], l['y1'], None])
                laser_z.extend([l['z0'], l['z1'], None])
                
            for r_slug in entry['railguns']:
                rx_h, ry_h, rz_h = get_railgun_helix(r_slug['x0'], r_slug['y0'], r_slug['z0'], r_slug['x1'], r_slug['y1'], r_slug['z1'])
                railgun_x.extend(rx_h + [None])
                railgun_y.extend(ry_h + [None])
                railgun_z.extend(rz_h + [None])

        pat_x, pat_y, pat_z, pat_txt = [], [], [], []
        for cap in cap_results:
            hist = cap['history']
            entry_c = min(hist, key=lambda x: abs(x['time'] - t))
            if entry_c['status'] != 'withdrawn':
                if entry_c['status'] != 'destroyed' or (t - cap.get('death_time', 9999.0) <= 1.5):
                    pat_x.append(entry_c['tx'])
                    pat_y.append(entry_c['ty'])
                    pat_z.append(entry_c['tz'])
                    
                    cap_locked = False
                    for eng in eng_results:
                        e_entry = min(eng['history'], key=lambda x: abs(x['time'] - t))
                        if eng['iff'] == 0 and not e_entry['destroyed'] and e_entry['status'] != 'withdrawing':
                            cap_dist = math.sqrt((entry_c['tx'] - e_entry['tx'])**2 + (entry_c['ty'] - e_entry['ty'])**2 + (entry_c['tz'] - e_entry['tz'])**2)
                            if cap_dist < 22000.0:
                                cap_locked = True
                                lock_x.extend([e_entry['tx'], entry_c['tx'], None])
                                lock_y.extend([e_entry['ty'], entry_c['ty'], None])
                                lock_z.extend([e_entry['tz'], entry_c['tz'], None])
                    
                    c_label = f"⚠️ [LOCK] {cap['callsign']}" if cap_locked else cap['callsign']
                    pat_txt.append(c_label)
                    
                    if entry_c['status'] == 'destroyed' and entry_c['time'] <= t:
                        exp_x.append(entry_c['tx'])
                        exp_y.append(entry_c['ty'])
                        exp_z.append(entry_c['tz'])
                        exp_sz.append(int(12 + (t - entry_c['time']) * 14))

        cur_chatter = [msg for msg_time, msg in chatter_log if msg_time <= t]
        last_chatter = cur_chatter[-5:]
        chatter_text = wrap_text("<br>".join(last_chatter), 42)
        
        frames.append(go.Frame(
            data=[
                go.Scatter3d(x=h_x, y=h_y, z=h_z, text=h_txt, marker=dict(size=h_sz, symbol=h_sym)),
                go.Scatter3d(x=a_x, y=a_y, z=a_z, text=a_txt, marker=dict(size=a_sz, symbol=a_sym)),
                go.Scatter3d(
                    x=exp_x if exp_x else [None],
                    y=exp_y if exp_y else [None],
                    z=exp_z if exp_z else [None],
                    marker=dict(size=exp_sz if exp_sz else 1)
                ),
                go.Scatter3d(x=pat_x, y=pat_y, z=pat_z, text=pat_txt),
                go.Scatter3d(x=allied_trail_x, y=allied_trail_y, z=allied_trail_z),
                go.Scatter3d(x=enemy_trail_x, y=enemy_trail_y, z=enemy_trail_z),
                go.Scatter3d(x=missile_trail_x, y=missile_trail_y, z=missile_trail_z),
                go.Scatter3d(x=flare_x if flare_x else [None], y=flare_y if flare_y else [None], z=flare_z if flare_z else [None]),
                go.Scatter3d(x=chaff_x if chaff_x else [None], y=chaff_y if chaff_y else [None], z=chaff_z if chaff_z else [None]),
                go.Scatter3d(x=laser_x if laser_x else [None], y=laser_y if laser_y else [None], z=laser_z if laser_z else [None]),
                go.Scatter3d(x=railgun_x if railgun_x else [None], y=railgun_y if railgun_y else [None], z=railgun_z if railgun_z else [None]),
                go.Scatter3d(x=lock_x if lock_x else [None], y=lock_y if lock_y else [None], z=lock_z if lock_z else [None])
            ],
            layout=dict(
                annotations=[
                    dict(
                        text=chatter_text,
                        xref="paper", yref="paper",
                        x=0.02, y=0.02,
                        xanchor="left", yanchor="bottom",
                        showarrow=False,
                        align="left",
                        font=dict(size=9, color="#81e6d9", family="Courier New"),
                        bgcolor="rgba(10, 15, 25, 0.45)",
                        bordercolor="rgba(129, 230, 217, 0.25)",
                        borderwidth=1
                    )
                ]
            ),
            name=f'frame_{k}',
            traces=[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        ))

    fig.frames = frames

    initial_chatter = [msg for msg_time, msg in chatter_log if msg_time <= 0.0]
    initial_chatter_text = wrap_text("<br>".join(initial_chatter[-5:]), 42)

    fig.update_layout(
        title='AWACS 3D Tactical Air Defense: Simulated Dogfights & Giant Command Cruiser Engagements',
        scene=dict(
            xaxis=dict(title='X (meters)', range=[x_min, x_max]),
            yaxis=dict(title='Y (meters)', range=[y_min, y_max]),
            zaxis=dict(title='Altitude (meters)', range=[z_min, z_max]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5),
            camera=dict(
                eye=dict(x=0.40, y=0.40, z=0.20)
            )
        ),
        annotations=[
            dict(
                text=initial_chatter_text,
                xref="paper", yref="paper",
                x=0.02, y=0.02,
                xanchor="left", yanchor="bottom",
                showarrow=False,
                align="left",
                font=dict(size=9, color="#81e6d9", family="Courier New"),
                bgcolor="rgba(10, 15, 25, 0.45)",
                bordercolor="rgba(129, 230, 217, 0.25)",
                borderwidth=1
            )
        ],
        template='plotly_dark',
        updatemenus=[dict(
            type='buttons',
            x=0.1, y=0,
            buttons=[
                dict(label='Scramble Fleet', method='animate', args=[None, dict(frame=dict(duration=120, redraw=True), fromcurrent=True)]),
                dict(label='Cease Fire (Pause)', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
            ]
        )],
        sliders=[dict(
            steps=[dict(
                method='animate',
                args=[[f'frame_{k}'], dict(mode='immediate', frame=dict(duration=120, redraw=True), transition=dict(duration=0))],
                label=f'{times[k]:.1f}s'
            ) for k in range(num_frames)],
            transition=dict(duration=0),
            x=0.2, y=-0.05,
            currentvalue=dict(font=dict(size=12), prefix='Airspace Clock: ', visible=True, xanchor='right'),
            len=0.8
        )]
    )

    fig.show()
