import json
import copy

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
add_markdown("""# Ace Combat AWACS Simulation (Windows Version)
**Parallelism via Apache Spark, CUDA, and Machine Learning**

This notebook executes:
- **PySpark** for big data processing and Random Forest ML threat scoring
- **CUDA (C++)** for parallel missile trajectory math
- **Interactive 3D Visualization** via Plotly

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
add_code("""# Generate Synthetic Radar Data (1,000,000 bogeys)
!python scripts/data_gen.py 1000000
""")

# Cell 5
add_markdown("## 2. Spark MLlib - Threat Prioritization")
add_code("""# Initialize PySpark Session
import os
import sys

# Ensure Spark can find Python (Adjust path if needed or let Spark use system default)
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ['JAVA_HOME'] = "C:/Users/nyoma/scoop/apps/java17/current"

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder \\
    .appName("AWACS_Threat_Scoring") \\
    .config("spark.driver.memory", "4g") \\
    .getOrCreate()

print("Spark Session created successfully.")
""")

# Cell 6
add_code("""# Load Data
df = spark.read.csv("radar_data.csv", header=True, inferSchema=True)

from pyspark.sql.functions import col, sqrt, atan2, degrees, abs as pyspark_abs, when

# Feature 1: Distance to base
df = df.withColumn("distance", sqrt(col("x")**2 + col("y")**2))

# Feature 2: Heading Difference (Are they pointing at us?)
# Calculate angle_to_base = atan2(-y, -x) and convert to degrees 0-360
df = df.withColumn("angle_to_base", (degrees(atan2(-col("y"), -col("x"))) + 360) % 360)
df = df.withColumn("raw_diff", pyspark_abs(col("heading") - col("angle_to_base")))
df = df.withColumn("heading_diff", when(col("raw_diff") > 180, 360 - col("raw_diff")).otherwise(col("raw_diff")))

# Assemble features using our new, calculated math!
assembler = VectorAssembler(
    inputCols=["distance", "altitude", "velocity", "heading_diff"],
    outputCol="features"
)
data = assembler.transform(df)
""")

# Cell 7
add_code("""# Train Random Forest Model
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

rf = RandomForestClassifier(labelCol="threat_label", featuresCol="features", numTrees=20)
print("Training Random Forest Model (Distributed)...")
model = rf.fit(train_data)

# Evaluate
predictions = model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(labelCol="threat_label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
""")

# Cell 8
add_code("""# Filter the most critical threats (Prediction == 3)
critical_df = predictions.filter(col("prediction") == 3.0)
top_threats = critical_df.orderBy("distance").limit(1000).toPandas()

print(f"Found {len(top_threats)} critical targets.")
top_threats.head()
""")

# Cell 9
add_markdown("## 3. CUDA - Parallel Interception Trajectories")
add_code("""# Note: Run build_cuda.bat in your terminal to compile the DLL first!
print("Ensure you have run build_cuda.bat to compile trajectory.dll")
""")

# Cell 10
add_code("""# Execute CUDA via ctypes
import ctypes
import numpy as np

# Load the shared library (DLL on Windows)
lib = ctypes.CDLL('./trajectory.dll')

# Define argument types
lib.calculate_interception.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int) # Evasions counter
]

num_targets = len(top_threats)
x_arr = np.array(top_threats['x'], dtype=np.float32)
y_arr = np.array(top_threats['y'], dtype=np.float32)
z_arr = np.array(top_threats['altitude'], dtype=np.float32)
v_arr = np.array(top_threats['velocity'], dtype=np.float32)
h_arr = np.array(top_threats['heading'], dtype=np.float32)

tti = np.zeros(num_targets, dtype=np.float32)
int_x = np.zeros(num_targets, dtype=np.float32)
int_y = np.zeros(num_targets, dtype=np.float32)
int_z = np.zeros(num_targets, dtype=np.float32)
evasions = ctypes.c_int(0)

print(f"Sending {num_targets} targets to CUDA GPU...")
lib.calculate_interception(
    num_targets,
    x_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    y_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    z_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    v_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    h_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    tti.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    int_x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    int_y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    int_z.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    ctypes.byref(evasions)
)

top_threats['tti'] = tti
top_threats['int_x'] = int_x
top_threats['int_y'] = int_y
top_threats['int_z'] = int_z

print("CUDA computation complete!")
print(f"CRITICAL ALERT: {evasions.value} targets successfully evaded our SAM network!")
""")

# Cell 11
add_markdown("## 4. AWACS Callouts & Visualization")
add_code("""# Generate AWACS Callouts for the top 5 threats
import math

def get_clock_position(x, y):
    angle_rad = math.atan2(x, y) 
    angle_deg = (math.degrees(angle_rad) + 360) % 360
    clock = int(round(angle_deg / 30.0))
    if clock == 0:
        clock = 12
    return clock

def get_elevation(z):
    if z > 10000: return "high"
    elif z < 3000: return "low"
    else: return "level"

print("\\n===== AWACS ALERTS =====")
for i, row in top_threats.head(5).iterrows():
    clock = get_clock_position(row['x'], row['y'])
    elevation = get_elevation(row['altitude'])
    print(f"AWACS: \\"Bogey, {clock} o'clock, {elevation}! Target ID {int(row['bogey_id'])}, distance {row['distance']/1000:.1f} km.\\"")
print("========================\\n")
""")

# Cell 12
add_code("""# Interactive 3D Visualization with Plotly
import plotly.graph_objects as go
import numpy as np

# Plot the Base
fig = go.Figure(data=[go.Scatter3d(
    x=[0], y=[0], z=[0],
    mode='markers',
    marker=dict(size=10, color='green', symbol='diamond'),
    name='Allied Base'
)])

# Generate Radar Dome (Sphere)
theta = np.linspace(0, 2.*np.pi, 50)
phi = np.linspace(0, np.pi/2, 50) # Hemisphere (only above ground)
theta, phi = np.meshgrid(theta, phi)
r = 100000 # 100km Radar Range

x_dome = r * np.sin(phi) * np.cos(theta)
y_dome = r * np.sin(phi) * np.sin(theta)
z_dome = r * np.cos(phi)

fig.add_trace(go.Surface(
    x=x_dome, y=y_dome, z=z_dome,
    opacity=0.1,
    colorscale=[[0, 'rgba(0,255,0,0.1)'], [1, 'rgba(0,255,0,0.1)']],
    showscale=False,
    name='Radar Dome (100km)'
))

# Plot Critical Bogeys
fig.add_trace(go.Scatter3d(
    x=top_threats['x'], y=top_threats['y'], z=top_threats['altitude'],
    mode='markers',
    marker=dict(size=3, color='red'),
    name='Critical Bogeys'
))

# Plot the top 5 Interception Paths
for i, row in top_threats.head(5).iterrows():
    if row['tti'] > 0:
        fig.add_trace(go.Scatter3d(
            x=[0, row['int_x']], y=[0, row['int_y']], z=[0, row['int_z']],
            mode='lines',
            line=dict(color='yellow', width=2, dash='dash'),
            name=f'Missile Trajectory {int(row["bogey_id"])}'
        ))

fig.update_layout(
    title='AWACS Radar Space & SAM Interception Trajectories',
    scene=dict(
        xaxis_title='X (meters)',
        yaxis_title='Y (meters)',
        zaxis_title='Altitude (meters)',
        aspectmode='data'
    ),
    template='plotly_dark'
)

fig.show()
""")

with open('windows_notebook.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("windows_notebook.ipynb created successfully.")
