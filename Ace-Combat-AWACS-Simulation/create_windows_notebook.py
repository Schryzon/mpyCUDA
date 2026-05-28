import json

notebook = {
  "cells": [],
  "metadata": {},
  "nbformat": 4,
  "nbformat_minor": 5
}

def add_markdown(source):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    })

def add_code(source):
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [line + "\n" for line in source.split("\n")]
    })

# Cell 1
add_markdown("""# Simulated Ace Combat: AWACS Tactical Air Defense (Windows Version)
**Parallelism via Apache Spark, CUDA, and Machine Learning**

Welcome to the tactical air defense simulation room. This notebook runs an end-to-end mission pipeline:
- **PySpark**: Distributed processing of 1 million radar logs, tracking proximity and vector heading relative to three tactical bases, and a Vector-based Random Forest model to distinguish hostile MiG/Sukhois from friendly F-15/F-14 CAP patrols.
- **CUDA (C++)**: GPGPU parallel intercept calculations using a custom quadratic solver that dynamically matches allied fighter speeds to specific bogey threats and intercepts from multiple bases in parallel.
- **Plotly 3D Tactical Map**: A beautiful, interactive, and fully animated tactical air space showing fighter scrambles, SAM defense launches, intercept dogfights, and dynamic hit explosions!

Project: **Schryzon/mpyCUDA**  
Course: **Parallel Processing A**
""")

# Cell 2
add_markdown("""> ### 🛑 PREREQUISITES FOR WINDOWS
> **1. NVIDIA GPU + CUDA Toolkit installed**
> **2. Visual Studio C++ Build Tools installed (for nvcc)**
> **3. Java 8+ installed (for PySpark)**
""")

# Cell 3
add_markdown("## 1. Environment Setup & Data Generation")
add_code("""# Ensure packages are installed
# !pip install pyspark plotly pandas numpy
""")

# Cell 4
add_code("""# Generate Synthetic Radar Data (1,000,000 aircraft records)
# Includes Allies (F-15, F-14) and Hostiles (MiG-29, Su-27) across 3 tactical bases.
!python scripts/data_gen.py 1000000
""")

# Cell 5
add_markdown("## 2. Spark MLlib - Multi-Base Threat Prioritization")
add_code("""# Initialize PySpark Session
import os
import sys

# Ensure Spark can find Python (Adjust path if needed or let Spark use system default)
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ['JAVA_HOME'] = "C:/Users/nyoma/scoop/apps/temurin17-jdk/current"
os.environ['SPARK_LOCAL_IP'] = "127.0.0.1"

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder \
    .appName("AWACS_Threat_Scoring") \
    .config("spark.driver.memory", "4g") \
    .config("spark.driver.host", "127.0.0.1") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .config("spark.local.dir", "C:/Windows/Temp") \
    .getOrCreate()

print("Spark Session created successfully.")
""")

# Cell 6
add_code("""# Load Data
df = spark.read.csv("radar_data.csv", header=True, inferSchema=True)

from pyspark.sql.functions import col, sqrt, atan2, degrees, abs as pyspark_abs, when, least

# 1. Define Tactical Base coordinates
# Base Alpha: (0, 0, 0)
# Base Bravo: (-60000, 40000, 0)
# Base Charlie: (50000, -50000, 0)

# 2. Feature Engineering: Calculate distances to each of our 3 bases
df = df.withColumn("dist_alpha", sqrt(col("x")**2 + col("y")**2 + col("altitude")**2))
df = df.withColumn("dist_bravo", sqrt((col("x") - (-60000.0))**2 + (col("y") - 40000.0)**2 + col("altitude")**2))
df = df.withColumn("dist_charlie", sqrt((col("x") - 50000.0)**2 + (col("y") - (-50000.0))**2 + col("altitude")**2))

# 3. Feature Engineering: Find minimum distance to the closest base
df = df.withColumn("distance", least("dist_alpha", "dist_bravo", "dist_charlie"))

# 4. Feature Engineering: Get the coordinates of the closest base
df = df.withColumn("bx_closest", 
    when(col("dist_alpha") <= col("dist_bravo"), 
        when(col("dist_alpha") <= col("dist_charlie"), 0.0).otherwise(50000.0)
    ).otherwise(
        when(col("dist_bravo") <= col("dist_charlie"), -60000.0).otherwise(50000.0)
    )
)
df = df.withColumn("by_closest", 
    when(col("dist_alpha") <= col("dist_bravo"), 
        when(col("dist_alpha") <= col("dist_charlie"), 0.0).otherwise(-50000.0)
    ).otherwise(
        when(col("dist_bravo") <= col("dist_charlie"), 40000.0).otherwise(-50000.0)
    )
)

# 5. Feature Engineering: Shortest angular heading difference relative to closest base
df = df.withColumn("angle_to_base", (degrees(atan2(col("by_closest") - col("y"), col("bx_closest") - col("x"))) + 360) % 360)
df = df.withColumn("raw_diff", pyspark_abs(col("heading") - col("angle_to_base")))
df = df.withColumn("heading_diff", when(col("raw_diff") > 180, 360 - col("raw_diff")).otherwise(col("raw_diff")))

# Assemble features incorporating IFF code and Aircraft Type ID for high-accuracy threat classification!
assembler = VectorAssembler(
    inputCols=["iff_status", "aircraft_type_id", "distance", "altitude", "velocity", "heading_diff"],
    outputCol="features"
)
data = assembler.transform(df)
""")

# Cell 7
add_code("""# Train Random Forest Model
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

rf = RandomForestClassifier(labelCol="threat_label", featuresCol="features", numTrees=20)
print("Training Multi-Base Threat Classification Model (Distributed)...")
model = rf.fit(train_data)

# Evaluate
predictions = model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(labelCol="threat_label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Model Accuracy (boosted by IFF + Closest Base tracking): {accuracy * 100:.2f}%")
""")

# Cell 8
add_code("""# Filter critical threats (Prediction == 3 and Hostile IFF == 0)
# We prioritize verified elite enemy squadrons (Grabacr, Strigon, Sol, Yellow, Gelb, Ofnir)
# over standard unnamed threats to ensure our AWACS scrambles target high-value bandits first,
# then we order by distance to our bases!
from pyspark.sql.functions import desc

critical_df = predictions.filter((col("prediction") == 3.0) & (col("iff_status") == 0))

# We order by whether the bogey belongs to an elite squadron first (descending), then by proximity!
top_threats = critical_df.orderBy(
    when(col("squadron_name") != "None", 1).otherwise(0).desc(),
    "distance"
).limit(1000).toPandas()

print(f"Found {len(top_threats)} critical targets.")
top_threats[['bogey_id', 'aircraft_type', 'squadron_name', 'callsign', 'distance', 'velocity', 'threat_label']].head()
""")

# Cell 9
add_markdown("## 3. CUDA - Parallel Interception Trajectories (Multi-Base Dogfights)")
add_code("""# Note: Run build_cuda.bat in your terminal to compile the DLL first!
print("Ensure you have run build_cuda.bat to compile trajectory.dll")
""")

# Cell 10
add_code("""# Execute CUDA via ctypes
import ctypes
import numpy as np

# Load the shared library (DLL on Windows)
lib = ctypes.CDLL('./trajectory.dll')

# Define argument types matching our upgraded multi-base aircraft targeting kernel
lib.calculate_interception.argtypes = [
    ctypes.c_int,                                            # num_targets
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),  # x, y, z
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),  # velocity, heading
    ctypes.POINTER(ctypes.c_int),                            # aircraft_type_id
    ctypes.c_int,                                            # num_bases
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),  # base_x, base_y, base_z
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),  # tti, int_x, int_y, int_z
    ctypes.POINTER(ctypes.c_int),                            # launch_base_idx
    ctypes.POINTER(ctypes.c_int)                             # evasions
]

num_targets = len(top_threats)
x_arr = np.array(top_threats['x'], dtype=np.float32)
y_arr = np.array(top_threats['y'], dtype=np.float32)
z_arr = np.array(top_threats['altitude'], dtype=np.float32)
v_arr = np.array(top_threats['velocity'], dtype=np.float32)
h_arr = np.array(top_threats['heading'], dtype=np.float32)
type_id_arr = np.array(top_threats['aircraft_type_id'], dtype=np.int32)

# Define Base locations
base_x_arr = np.array([0.0, -60000.0, 50000.0], dtype=np.float32)
base_y_arr = np.array([0.0, 40000.0, -50000.0], dtype=np.float32)
base_z_arr = np.array([0.0, 0.0, 0.0], dtype=np.float32)
num_bases = len(base_x_arr)

# Buffers for results
tti = np.zeros(num_targets, dtype=np.float32)
int_x = np.zeros(num_targets, dtype=np.float32)
int_y = np.zeros(num_targets, dtype=np.float32)
int_z = np.zeros(num_targets, dtype=np.float32)
launch_base_idx = np.zeros(num_targets, dtype=np.int32)
evasions = ctypes.c_int(0)

print(f"Sending {num_targets} targets to CUDA GPU...")
lib.calculate_interception(
    num_targets,
    x_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    y_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    z_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    v_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    h_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    type_id_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    num_bases,
    base_x_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    base_y_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    base_z_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    tti.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    int_x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    int_y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    int_z.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    launch_base_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    ctypes.byref(evasions)
)

top_threats['tti'] = tti
top_threats['int_x'] = int_x
top_threats['int_y'] = int_y
top_threats['int_z'] = int_z
top_threats['launch_base_idx'] = launch_base_idx

print("CUDA computation complete!")
print(f"CRITICAL ALERT: {evasions.value} hostiles successfully evaded our airspace defense network!")
top_threats[['bogey_id', 'aircraft_type', 'squadron_name', 'callsign', 'distance', 'tti', 'launch_base_idx']].head()
""")

# Cell 11
add_markdown("## 4. AWACS Tactical Calls & Visual dogfights")
add_code("""# Generate AWACS Callouts for the top 5 threats
import math

# Coordinate offsets for clock position mapping relative to closest base
def get_clock_position(x, y, bx, by):
    dx, dy = x - bx, y - by
    angle_rad = math.atan2(dx, dy) 
    angle_deg = (90.0 - math.degrees(angle_rad) + 360.0) % 360.0
    clock = int(round(angle_deg / 30.0))
    if clock == 0: clock = 12
    return clock

def get_elevation(z):
    if z > 10000: return "high"
    elif z < 3000: return "low"
    else: return "level"

base_names = {0: "Alpha (HQ)", 1: "Bravo (FOB)", 2: "Charlie (Naval)"}

matchup_dict = {
    0: "F-15 EAGLE",
    1: "F-14 TOMCAT",
    9: "F-22A MOBIUS (Raptor)",
    10: "F-15C GALM (Eagle)",
    11: "F-15E GARUDA (Strike Eagle)",
    12: "F-14D WARDOG (Tomcat)",
    13: "F-16C CROW (Falcon)"
}

print("\\n===== AWACS RADAR CALLOUTS =====")
for i, row in top_threats.head(5).iterrows():
    b_idx = int(row['launch_base_idx'])
    if b_idx >= 0:
        bx, by = base_x_arr[b_idx], base_y_arr[b_idx]
        b_name = base_names.get(b_idx, "Unknown")
    else:
        # Threat evaded. Use closest base coordinates and name for warning.
        bx, by = row['bx_closest'], row['by_closest']
        if abs(bx - 0.0) < 1.0:
            b_name = "Alpha (HQ)"
        elif abs(bx - (-60000.0)) < 1.0:
            b_name = "Bravo (FOB)"
        else:
            b_name = "Charlie (Naval)"
            
    clock = get_clock_position(row['x'], row['y'], bx, by)
    elevation = get_elevation(row['altitude'])
    
    callsign_str = f"{row['callsign']} ({row['aircraft_type']})" if row['squadron_name'] != 'None' else row['aircraft_type']
    
    if row['tti'] > 0:
        matchup = matchup_dict.get(row['aircraft_type_id'], "SAM MISSILE")
        print(f"AWACS: \\"Bandit {callsign_str}, hot at {clock} o'clock, {elevation}! Base {b_name} scrambling {matchup} interceptor. TTI {row['tti']:.1f}s!\\"")
    else:
        print(f"AWACS: \\"WARNING! High-speed Bandit {callsign_str}, {clock} o'clock, {elevation} relative to Base {b_name} has breached intercept envelope!\\"")
print("================================\\n")
""")

# Cell 12
add_code("""# Interactive 3D Animated Tactical Map with Plotly
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import math

print("Assembling 3D tactical dogfight animation frames...")

# 1. Base stations setup (bases are squares as requested!)
fig = go.Figure(data=[
    go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=12, color='lime', symbol='square'), name='Base Alpha (HQ)'),
    go.Scatter3d(x=[-60000], y=[40000], z=[0], mode='markers', marker=dict(size=12, color='cyan', symbol='square'), name='Base Bravo (FOB)'),
    go.Scatter3d(x=[50000], y=[-50000], z=[0], mode='markers', marker=dict(size=12, color='magenta', symbol='square'), name='Base Charlie (Naval)')
])

# 2. Semi-translucent Radar Domes for each base (using Mesh3d for a smooth glass-like dome!)
theta = np.linspace(0, 2.*np.pi, 30)
phi = np.linspace(0, np.pi/2, 15)
theta, phi = np.meshgrid(theta, phi)
r = 100000 # 100km dome range

x_d = (r * np.sin(phi) * np.cos(theta)).flatten()
y_d = (r * np.sin(phi) * np.sin(theta)).flatten()
z_d = (r * np.cos(phi)).flatten()

# Add Mesh3d domes (glass-like, no spiral grids)
fig.add_trace(go.Mesh3d(x=x_d, y=y_d, z=z_d, opacity=0.03, color='lime', alphahull=0, name='Dome Alpha (HQ)', showlegend=True))
fig.add_trace(go.Mesh3d(x=x_d - 60000, y=y_d + 40000, z=z_d, opacity=0.03, color='cyan', alphahull=0, name='Dome Bravo (FOB)', showlegend=True))
fig.add_trace(go.Mesh3d(x=x_d + 50000, y=y_d - 50000, z=z_d, opacity=0.03, color='magenta', alphahull=0, name='Dome Charlie (Naval)', showlegend=True))

# 3. Static background threat plots
fig.add_trace(go.Scatter3d(
    x=top_threats['x'], y=top_threats['y'], z=top_threats['altitude'],
    mode='markers', marker=dict(size=2, color='rgba(255, 0, 0, 0.15)'),
    name='Threat Radar Blips'
))

# 4. Grab top engagements to animate. Let's make sure we select our elite squadrons!
elite_engagements = top_threats[(top_threats['tti'] > 0) & (top_threats['squadron_name'] != 'None')].head(20)
if len(elite_engagements) < 15:
    standard_engagements = top_threats[(top_threats['tti'] > 0) & (top_threats['squadron_name'] == 'None')].head(20 - len(elite_engagements))
    engagements = pd.concat([elite_engagements, standard_engagements]).copy()
else:
    engagements = elite_engagements.copy()

# Initial Hostile Positions (t = 0) (Red diamonds represent enemy 3D triangle wings!)
fig.add_trace(go.Scatter3d(
    x=engagements['x'], y=engagements['y'], z=engagements['altitude'],
    mode='markers+text', marker=dict(size=8, color='red', symbol='diamond'),
    text=engagements.apply(lambda r: f"{r['callsign']} ({r['aircraft_type']})" if r['squadron_name'] != 'None' else r['aircraft_type'], axis=1), 
    textposition='top center',
    name='Hostiles (MiG/Su)'
))

# Initial Interceptor Positions (t = 0, starting at their launch bases) (Cyan diamonds represent allied 3D triangle wings!)
bx_coords, by_coords, bz_coords, interceptor_names = [], [], [], []
for _, row in engagements.iterrows():
    b_idx = int(row['launch_base_idx'])
    bx_coords.append(base_x_arr[b_idx])
    by_coords.append(base_y_arr[b_idx])
    bz_coords.append(base_z_arr[b_idx])
    
    # Dynamic allied fighter types matching
    t_type = int(row['aircraft_type_id'])
    if t_type == 9: interceptor_names.append('Mobius Squadron (F-22A)')
    elif t_type == 10: interceptor_names.append('Galm Team (F-15C)')
    elif t_type == 11: interceptor_names.append('Garuda Team (F-15E)')
    elif t_type == 12: interceptor_names.append('Wardog Squadron (F-14D)')
    elif t_type == 13: interceptor_names.append('Crow Team (F-16C)')
    elif t_type == 0: interceptor_names.append('Allied F-15 Eagle')
    elif t_type == 1: interceptor_names.append('Allied F-14 Tomcat')
    else: interceptor_names.append('SAM Battery')

fig.add_trace(go.Scatter3d(
    x=bx_coords, y=by_coords, z=bz_coords,
    mode='markers+text', marker=dict(size=8, color='cyan', symbol='diamond'),
    text=interceptor_names, textposition='bottom center',
    name='Allied Scrambles'
))

# Hit Explosion Markers (t = 0, empty)
fig.add_trace(go.Scatter3d(
    x=[None], y=[None], z=[None],
    mode='markers', marker=dict(size=1, color='orange', symbol='circle'),
    name='Dogfight Hits'
))

# Allied Combat Air Patrols (CAP) patrolling the airspace in Finger-Four/V formations!
patrol_x, patrol_y, patrol_z = [], [], []
cap_formations = {
    0: [ # Mobius (F-22A) - V-Formation
        (0.0, 0.0, 0.0),
        (-1200.0, -1200.0, 100.0),
        (1200.0, -1200.0, -100.0),
        (-2400.0, -2400.0, 200.0)
    ],
    1: [ # Wardog (F-14D) - Finger-Four
        (0.0, 0.0, 0.0),
        (-1200.0, -1200.0, 50.0),
        (2400.0, -600.0, -50.0),
        (1200.0, -1800.0, 100.0)
    ],
    2: [ # Razgriz (F-14D) - Finger-Four
        (0.0, 0.0, 0.0),
        (-1200.0, -1200.0, -50.0),
        (2400.0, -600.0, 50.0),
        (1200.0, -1800.0, -100.0)
    ]
}

# Initial CAP positions (t = 0)
for b_idx in range(num_bases):
    bx, by = base_x_arr[b_idx], base_y_arr[b_idx]
    lead_x = bx + 20000.0
    lead_y = by
    lead_z = 10000.0
    
    # Heading is tangent to the circle path (90 degrees, i.e., East/North)
    for dx, dy, dz in cap_formations[b_idx]:
        patrol_x.append(lead_x + dx)
        patrol_y.append(lead_y + dy)
        patrol_z.append(lead_z + dz)

fig.add_trace(go.Scatter3d(
    x=patrol_x, y=patrol_y, z=patrol_z,
    mode='markers', marker=dict(size=7, color='rgba(0, 255, 200, 0.7)', symbol='diamond'),
    name='CAP Patrol Squads'
))

# 5. Continuous Line Trails & Missile Launch Shoots
# Trace 11: Allied Trails (green)
fig.add_trace(go.Scatter3d(
    x=[None], y=[None], z=[None],
    mode='lines', line=dict(color='rgba(0, 255, 100, 0.6)', width=2),
    name='Allied Scramble Trails'
))

# Trace 12: Enemy Trails (orange)
fig.add_trace(go.Scatter3d(
    x=[None], y=[None], z=[None],
    mode='lines', line=dict(color='rgba(255, 120, 0, 0.6)', width=2),
    name='Enemy Flight Trails'
))

# Trace 13: Missile Shoot Lines (cyan dashed)
fig.add_trace(go.Scatter3d(
    x=[None], y=[None], z=[None],
    mode='lines', line=dict(color='rgba(0, 255, 255, 0.8)', width=1.5, dash='dash'),
    name='Missile Launch Vectors'
))

# Trace Index mappings:
# 0,1,2: Bases
# 3,4,5: Domes
# 6: Threat Context
# 7: Animated Hostiles
# 8: Animated Allied Scrambles
# 9: Animated Hits
# 10: CAP Patrol Squads
# 11: Allied Trails
# 12: Enemy Trails
# 13: Missile Shoot Lines

# Calculate Animation Frames
num_frames = 60
max_time = float(engagements['tti'].max() * 1.1)
times = np.linspace(0, max_time, num_frames)
dt = times[1] - times[0]

frames = []
for k, t in enumerate(times):
    b_x, b_y, b_z, b_txt = [], [], [], []
    i_x, i_y, i_z, i_txt = [], [], [], []
    exp_x, exp_y, exp_z, exp_sz = [], [], [], []
    
    # Trails and missile coordinates lists
    allied_trail_x, allied_trail_y, allied_trail_z = [], [], []
    enemy_trail_x, enemy_trail_y, enemy_trail_z = [], [], []
    missile_x, missile_y, missile_z = [], [], []
    
    for _, row in engagements.iterrows():
        px, py, pz = row['x'], row['y'], row['altitude']
        h_rad = row['heading'] * math.pi / 180.0
        vx = row['velocity'] * math.sin(h_rad)
        vy = row['velocity'] * math.cos(h_rad)
        tti_val = row['tti']
        b_idx = int(row['launch_base_idx'])
        bx, by, bz = base_x_arr[b_idx], base_y_arr[b_idx], base_z_arr[b_idx]
        
        # Interceptor classification
        t_type = int(row['aircraft_type_id'])
        int_type = 'SAM Missile'
        if t_type == 9: int_type = 'Mobius Squadron (F-22A)'
        elif t_type == 10: int_type = 'Galm Team (F-15C)'
        elif t_type == 11: int_type = 'Garuda Team (F-15E)'
        elif t_type == 12: int_type = 'Wardog Squadron (F-14D)'
        elif t_type == 13: int_type = 'Crow Team (F-16C)'
        elif t_type == 0: int_type = 'Allied F-15 Eagle'
        elif t_type == 1: int_type = 'Allied F-14 Tomcat'
        
        # Positions
        hx_t = px + vx * min(t, tti_val)
        hy_t = py + vy * min(t, tti_val)
        hz_t = pz
        
        ratio = min(t, tti_val) / tti_val
        ax_t = bx + (row['int_x'] - bx) * ratio
        ay_t = by + (row['int_y'] - by) * ratio
        az_t = bz + (row['int_z'] - bz) * ratio
        
        # Enemy flight path trail
        enemy_trail_x.extend([px, hx_t, None])
        enemy_trail_y.extend([py, hy_t, None])
        enemy_trail_z.extend([pz, hz_t, None])
        
        # Allied scramble path trail
        allied_trail_x.extend([bx, ax_t, None])
        allied_trail_y.extend([by, ay_t, None])
        allied_trail_z.extend([bz, az_t, None])
        
        # Missile shoot trajectory (Launched at 70% of TTI)
        t_launch = tti_val * 0.7
        if t >= t_launch:
            # Fighter launch position
            ax_launch = bx + (row['int_x'] - bx) * 0.7
            ay_launch = by + (row['int_y'] - by) * 0.7
            az_launch = bz + (row['int_z'] - bz) * 0.7
            
            m_ratio = min((t - t_launch) / (tti_val - t_launch), 1.0)
            mx_t = ax_launch + (row['int_x'] - ax_launch) * m_ratio
            my_t = ay_launch + (row['int_y'] - ay_launch) * m_ratio
            mz_t = az_launch + (row['int_z'] - az_launch) * m_ratio
            
            # Connect launch site to missile head
            missile_x.extend([ax_launch, mx_t, None])
            missile_y.extend([ay_launch, my_t, None])
            missile_z.extend([az_launch, mz_t, None])
            
        if t < tti_val:
            b_x.append(hx_t)
            b_y.append(hy_t)
            b_z.append(hz_t)
            b_txt.append(f"{row['callsign']} ({row['aircraft_type']})" if row['squadron_name'] != 'None' else row['aircraft_type'])
            
            i_x.append(ax_t)
            i_y.append(ay_t)
            i_z.append(az_t)
            i_txt.append(int_type)
        else:
            if t < tti_val + 2.5:
                exp_x.append(row['int_x'])
                exp_y.append(row['int_y'])
                exp_z.append(row['int_z'])
                exp_sz.append(int(10 + (t - tti_val) * 12))
                
    # Allied CAP Flight positions orbiting in tight formations around their bases
    pat_x, pat_y, pat_z = [], [], []
    for b_idx in range(num_bases):
        bx, by = base_x_arr[b_idx], base_y_arr[b_idx]
        omega = 0.06 # orbit angular speed
        rad = 20000.0
        
        # Center of the formation
        lead_x = bx + rad * math.cos(omega * t + b_idx * math.pi/3.0)
        lead_y = by + rad * math.sin(omega * t + b_idx * math.pi/3.0)
        lead_z = 10000.0 + math.sin(omega * t) * 1200.0
        
        # Compute heading tangent to circle
        h_rad = omega * t + b_idx * math.pi/3.0 + math.pi/2.0
        c = math.cos(h_rad)
        s = math.sin(h_rad)
        
        for dx, dy, dz in cap_formations[b_idx]:
            rx = dx * c + dy * s
            ry = -dx * s + dy * c
            pat_x.append(lead_x + rx)
            pat_y.append(lead_y + ry)
            pat_z.append(lead_z + dz)
        
    frames.append(go.Frame(
        data=[
            go.Scatter3d(x=b_x, y=b_y, z=b_z, text=b_txt, marker=dict(size=8, color='red', symbol='diamond')),
            go.Scatter3d(x=i_x, y=i_y, z=i_z, text=i_txt, marker=dict(size=8, color='cyan', symbol='diamond')),
            go.Scatter3d(
                x=exp_x if len(exp_x) > 0 else [None],
                y=exp_y if len(exp_y) > 0 else [None],
                z=exp_z if len(exp_z) > 0 else [None],
                marker=dict(size=exp_sz if len(exp_sz) > 0 else 1, color='orange', symbol='circle')
            ),
            go.Scatter3d(x=pat_x, y=pat_y, z=pat_z, marker=dict(size=7, color='rgba(0, 255, 200, 0.7)', symbol='diamond')),
            go.Scatter3d(x=allied_trail_x, y=allied_trail_y, z=allied_trail_z),
            go.Scatter3d(x=enemy_trail_x, y=enemy_trail_y, z=enemy_trail_z),
            go.Scatter3d(x=missile_x, y=missile_y, z=missile_z)
        ],
        name=f'frame_{k}',
        traces=[7, 8, 9, 10, 11, 12, 13]
    ))

fig.frames = frames

# 6. UI layout controls: Play, Pause, and Timeline Slider
fig.update_layout(
    title='AWACS Tactical Battle Space: Simulated Ace Combat Scrambles',
    scene=dict(
        xaxis=dict(title='X (meters)', range=[-200000, 200000]),
        yaxis=dict(title='Y (meters)', range=[-200000, 200000]),
        zaxis=dict(title='Altitude (meters)', range=[0, 25000]),
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=0.5),
        camera=dict(
            eye=dict(x=0.4, y=0.4, z=0.2)
        )
    ),
    template='plotly_dark',
    updatemenus=[dict(
        type='buttons',
        x=0.1, y=0,
        buttons=[
            dict(label='Scramble Fighters', method='animate', args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)]),
            dict(label='Cease Fire (Pause)', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
        ]
    )],
    sliders=[dict(
        steps=[dict(
            method='animate',
            args=[[f'frame_{k}'], dict(mode='immediate', frame=dict(duration=100, redraw=True), transition=dict(duration=0))],
            label=f'{times[k]:.1f}s'
        ) for k in range(num_frames)],
        transition=dict(duration=0),
        x=0.2, y=-0.05,
        currentvalue=dict(font=dict(size=12), prefix='Airspace Clock: ', visible=True, xanchor='right'),
        len=0.8
    )]
)

fig.show()
""")

with open('windows_notebook.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("windows_notebook.ipynb created successfully.")
