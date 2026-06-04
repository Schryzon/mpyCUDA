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
add_markdown("""# Simulated Ace Combat: AWACS Tactical Air Defense (Google Colab Version)
**Parallelism via Apache Spark, CUDA, and Machine Learning**

Welcome to the tactical air defense simulation room. This notebook runs an end-to-end mission pipeline:
- **PySpark**: Distributed processing of 1 million radar logs, tracking proximity and vector heading relative to three tactical bases, and a Vector-based Random Forest model to distinguish hostile MiG/Sukhois from friendly F-15/F-14 CAP patrols.
- **CUDA (C++)**: GPGPU parallel intercept calculations using a custom quadratic solver that dynamically matches allied fighter speeds to specific bogey threats and intercepts from multiple bases in parallel.
- **Plotly 3D Tactical Map**: A beautiful, interactive, and fully animated tactical air space showing fighter scrambles, SAM defense launches, intercept dogfights, and dynamic hit explosions!

Project: **Schryzon/mpyCUDA**  
Course: **Parallel Processing A**
""")

# Cell 2
add_markdown("""> ### 🛑 GPU REQUIRED!
> **This project REQUIRES a CUDA-capable GPU (NVIDIA T4 or better).**
> Go to **Runtime** → **Change runtime type**, select **T4 GPU**, and click **Save**.
""")

# Cell 3
add_markdown("## 1. Environment Setup & Data Generation")
add_code("""# Mount Google Drive and Setup Repository
import os
from google.colab import drive

drive.mount('/content/drive')

DRIVE_PATH = '/content/drive/MyDrive/Jay-IF24-mpyCUDA'
REPO_URL   = 'https://github.com/Schryzon/mpyCUDA.git'

if not os.path.exists(DRIVE_PATH):
    !git clone "{REPO_URL}" "{DRIVE_PATH}"
else:
    !git -C "{DRIVE_PATH}" pull

WORK_DIR = '/content/mpyCUDA'
if not os.path.exists(WORK_DIR):
    !ln -s "{DRIVE_PATH}" "{WORK_DIR}"

%cd {WORK_DIR}/Ace-Combat-AWACS-Simulation
""")

# Cell 4
add_code("""!apt-get update -qq
import os

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"

!pip install pyspark plotly -q

!java -version
""")

# Cell 5
add_code("""# Generate Synthetic Radar Data (1,000,000 aircraft records)
# Includes Allies (F-15, F-14) and Hostiles (MiG-29, Su-27) across 3 tactical bases.
!python scripts/data_gen.py 1000000
""")

# Cell 6
add_markdown("## 2. Spark MLlib - Multi-Base Threat Prioritization")
add_code("""# Initialize PySpark Session
import os
import sys

# Override standard Temp directory to avoid PermissionError on Java gateway file reads
local_temp = os.path.abspath("./tmp")
os.makedirs(local_temp, exist_ok=True)
os.environ['TEMP'] = local_temp
os.environ['TMP'] = local_temp

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Start cleanly with memory-optimized driver bounds
spark = SparkSession.builder \
    .appName("AWACS_Threat_Scoring") \
    .config("spark.driver.memory", "10g") \
    .config("spark.driver.host", "127.0.0.1") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .config("spark.local.dir", os.path.join(local_temp, "spark-local")) \
    .getOrCreate()

print("Spark Session created successfully.")
""")

# Cell 7
add_code("""# Load Data
df = spark.read.csv("radar_data.csv", header=True, inferSchema=True)

from pyspark.sql.functions import col, sqrt, atan2, degrees, abs as pyspark_abs, when, least, cos, lit
import math

# 1. Define Tactical Base coordinates
# Base Alpha: (0, 0, 0)
# Base Bravo: (-60000, 40000, 0)
# Base Charlie: (50000, -50000, 0)

# 2. Feature Engineering: Calculate distances to each of our 3 bases in 2D (xy plane) matching ground truth score
df = df.withColumn("dist_alpha", sqrt(col("x")**2 + col("y")**2))
df = df.withColumn("dist_bravo", sqrt((col("x") - (-60000.0))**2 + (col("y") - 40000.0)**2))
df = df.withColumn("dist_charlie", sqrt((col("x") - 50000.0)**2 + (col("y") - (-50000.0))**2))

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

# 5. Feature Engineering: Shortest angular heading difference using navigation heading coordinate standards
df = df.withColumn("angle_to_base", (90.0 - degrees(atan2(col("by_closest") - col("y"), col("bx_closest") - col("x"))) + 360.0) % 360.0)
df = df.withColumn("raw_diff", pyspark_abs(col("heading") - col("angle_to_base")))
df = df.withColumn("heading_diff", when(col("raw_diff") > 180, 360 - col("raw_diff")).otherwise(col("raw_diff")))

# 6. Advanced Feature Engineering for High-Accuracy Threat Scoring
# Closing speed to the base: positive means flying towards base, negative means flying away
df = df.withColumn("closing_speed", col("velocity") * cos(col("heading_diff") * (math.pi / 180.0)))

# Estimated time to impact base (seconds). If moving away, use large sentinel 99999.0
df = df.withColumn("time_to_impact", when(col("closing_speed") > 0, col("distance") / col("closing_speed")).otherwise(99999.0))

# Speed-to-altitude ratio (captures low-altitude high-speed bogeys sneaking under defense nets)
df = df.withColumn("speed_altitude_ratio", col("velocity") / (col("altitude") + 1.0))
""")

# Cell 8
add_code("""# Train Improved Random Forest Model with Pipeline and Hierarchical System logic
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Split data
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Symmetrical System Design: allies are friendly (threat 0). We filter bogeys (iff_status == 0) to train the model
train_enemies = train_data.filter(col("iff_status") == 0)

# Define Pipeline Stages
# StringIndexer maps categorical features explicitly and registers them with metadata
aircraft_indexer = StringIndexer(inputCol="aircraft_type_id", outputCol="aircraft_indexed").setHandleInvalid("keep")

# Assemble engineered features (iff_indexed not needed since model is trained only on enemies)
feature_cols = [
    "aircraft_indexed", 
    "distance", 
    "altitude", 
    "velocity", 
    "heading_diff", 
    "closing_speed", 
    "time_to_impact", 
    "speed_altitude_ratio"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Tuned Random Forest Classifier for multiclass threat levels
rf = RandomForestClassifier(
    labelCol="threat_label", 
    featuresCol="features", 
    numTrees=50, 
    maxDepth=10,
    maxBins=64,
    impurity="gini",
    seed=42
)

# Pipeline encapsulates the entire preprocessing and fitting workflow
pipeline = Pipeline(stages=[aircraft_indexer, assembler, rf])

print("Training Improved Pipeline-indexed Random Forest Threat Model (Distributed PySpark)...")
pipeline_model = pipeline.fit(train_enemies)

# Predict on test data
test_predictions = pipeline_model.transform(test_data)

# Hierarchical Prediction step: allies (iff_status == 1) are deterministically threat 0
predictions = test_predictions.withColumn(
    "prediction",
    when(col("iff_status") == 1, 0.0).otherwise(col("prediction"))
)

# Compute detailed classification metrics
eval_acc = MulticlassClassificationEvaluator(labelCol="threat_label", predictionCol="prediction", metricName="accuracy")
eval_prec = MulticlassClassificationEvaluator(labelCol="threat_label", predictionCol="prediction", metricName="weightedPrecision")
eval_rec = MulticlassClassificationEvaluator(labelCol="threat_label", predictionCol="prediction", metricName="weightedRecall")
eval_f1 = MulticlassClassificationEvaluator(labelCol="threat_label", predictionCol="prediction", metricName="f1")

accuracy = eval_acc.evaluate(predictions)
precision = eval_prec.evaluate(predictions)
recall = eval_rec.evaluate(predictions)
f1_score = eval_f1.evaluate(predictions)

print("\\n===== Spark ML Model Classification Report =====")
print(f"Model Accuracy     : {accuracy * 100:.2f}%")
print(f"Weighted Precision : {precision * 100:.2f}%")
print(f"Weighted Recall    : {recall * 100:.2f}%")
print(f"Weighted F1-Score  : {f1_score * 100:.2f}%")
print("=================================================")
""")

# Cell 9
add_code("""# Data Visualization: Feature Correlation, Confusion Matrix, and Importances
from pyspark.ml.stat import Correlation
import plotly.express as px
import pandas as pd
import numpy as np

# 1. Compute correlation matrix on engineered numeric features
numeric_cols = ["distance", "altitude", "velocity", "heading_diff", "closing_speed", "time_to_impact", "speed_altitude_ratio"]
corr_assembler = VectorAssembler(inputCols=numeric_cols, outputCol="corr_features")
corr_df = corr_assembler.transform(df)

r1 = Correlation.corr(corr_df, "corr_features").head()
correlation_matrix = r1[0].toArray()
corr_matrix_df = pd.DataFrame(correlation_matrix, index=numeric_cols, columns=numeric_cols)

fig_corr = px.imshow(
    corr_matrix_df,
    text_auto=".2f",
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
    title="Feature Correlation Matrix Heatmap",
    template="plotly_dark"
)
fig_corr.update_layout(width=550, height=550, margin=dict(l=60, r=60, b=60, t=60))
fig_corr.show()

# 2. Compute and print confusion matrix
confusion_matrix = predictions.groupBy("threat_label").pivot("prediction", [0.0, 1.0, 2.0, 3.0]).count().na.fill(0).orderBy("threat_label")
print("\\n===== Spark Distributed Confusion Matrix =====")
confusion_matrix.show()

conf_df = confusion_matrix.toPandas().set_index("threat_label")
conf_df.columns = ["Low (Pred)", "Medium (Pred)", "High (Pred)", "Critical (Pred)"]
conf_df.index = ["Low (True)", "Medium (True)", "High (True)", "Critical (True)"]

fig_conf = px.imshow(
    conf_df,
    text_auto=True,
    color_continuous_scale="Teal",
    title="Threat Level Confusion Matrix Heatmap",
    template="plotly_dark"
)
fig_conf.update_layout(width=500, height=500, margin=dict(l=60, r=60, b=60, t=60))
fig_conf.show()

# 3. Analyze and plot Feature Importances
rf_model = pipeline_model.stages[-1]
importances = rf_model.featureImportances.toArray()
feature_names = ["Aircraft Type ID", "Min Base Distance", "Altitude", "Velocity", "Heading Diff", "Closing Speed", "Time-To-Impact", "Speed-Altitude Ratio"]

feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=True)

fig_imp = px.bar(
    feat_imp_df, 
    x='Importance', 
    y='Feature', 
    orientation='h',
    title='Threat Classification Model - Feature Importances',
    color='Importance',
    color_continuous_scale='Teal',
    template='plotly_dark'
)
fig_imp.update_layout(showlegend=False, height=450, margin=dict(l=60, r=60, b=60, t=60))
fig_imp.show()
""")

# Cell 10
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

# Cell 10
add_markdown("## 3. CUDA - Parallel Interception Trajectories (Multi-Base Dogfights)")
add_code("""# Compile the CUDA C++ kernel into a shared library on Linux
!nvcc -Xcompiler -fPIC -shared -o libtrajectory.so scripts/trajectory_math.cu
print("CUDA Kernel compiled to libtrajectory.so")
""")

# Cell 11
add_code("""# Execute CUDA via ctypes
import ctypes
import numpy as np

# Load the shared library
lib = ctypes.CDLL('./libtrajectory.so')

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

# Cell 12
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
    2: "F-15 EAGLE",
    3: "F-14 TOMCAT",
    4: "F-22A RAPTOR",
    9: "F-22A MOBIUS (Raptor)",
    10: "F-15C GALM (Eagle)",
    11: "F-15E GARUDA (Strike Eagle)",
    12: "F-14D WARDOG (Tomcat)",
    13: "F-16C CROW (Falcon)",
    14: "ADF-01 FALKEN (TLS Laser)",
    15: "X-02S STRIKE WYVERN (EML Railgun)",
    16: "ADF-01 FALKEN (TLS Laser)",
    17: "X-02S STRIKE WYVERN (EML Railgun)",
    18: "PW-Mk.I (EML Railgun)",
    19: "X-02S STRIKE WYVERN (EML Railgun)"
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

# Cell 13
add_code("""# Interactive 3D Animated Tactical Map with Plotly
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import math
import random

print("Assembling 3D tactical dogfight animation frames using advanced physical simulation...")

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

# 4. Grab top engagements to animate.
# Select 15 hostile elite engagements first, fill with standard bogeys if needed
elite_engagements = top_threats[(top_threats['tti'] > 0) & (top_threats['squadron_name'] != 'None')].head(15)
if len(elite_engagements) < 12:
    standard_engagements = top_threats[(top_threats['tti'] > 0) & (top_threats['squadron_name'] == 'None')].head(12 - len(elite_engagements))
    engagements = pd.concat([elite_engagements, standard_engagements]).copy()
else:
    engagements = elite_engagements.copy()

# Query PySpark DataFrame for 5 elite allied squads to show patrolling / engaging
try:
    allied_elites = predictions.filter((col("iff_status") == 1) & (col("squadron_name") != 'None')).limit(5).toPandas()
    if not allied_elites.empty:
        allied_elites['tti'] = 25.0
        allied_elites['launch_base_idx'] = 0
        allied_elites['int_x'] = allied_elites['x'] + 10000.0
        allied_elites['int_y'] = allied_elites['y'] + 10000.0
        allied_elites['int_z'] = allied_elites['altitude']
        engagements = pd.concat([engagements, allied_elites]).copy()
except Exception as e:
    print("Could not query allied elites, sticking to hostiles only.")

engagements = engagements.reset_index(drop=True)

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
            is_airship = (t_type == 18 or t_type == 19)
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
                elif t_type == 14: i_name, i_weapon, i_speed = "FALKEN (TLS)", "laser", 900.0
                elif t_type == 15: i_name, i_weapon, i_speed = "WYVERN (EML)", "railgun", 850.0
                elif t_type == 16: i_name, i_weapon, i_speed = "FALKEN (TLS)", "laser", 900.0
                elif t_type == 17: i_name, i_weapon, i_speed = "WYVERN (EML)", "railgun", 850.0
                elif t_type == 19: i_name, i_weapon, i_speed = "WYVERN (EML)", "railgun", 850.0
            else:
                # Allied target -> Enemy interceptor
                i_name = "Hostile Interceptor"
                if t_type == 14: i_name, i_weapon, i_speed = "RAVEN (TLS)", "laser", 900.0
                elif t_type == 15: i_name, i_weapon, i_speed = "PW-Mk.I (EML)", "railgun", 850.0
                elif t_type == 18: i_name, i_weapon, i_speed = "PW-Mk.I (EML)", "railgun", 850.0
                else: i_name, i_speed = "Hostile MiG-29", 680.0
                
            # Interceptor starts at base
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
            global_chatter.append((0.0, f"AWACS: Scramble order issued from Base {b_name}! {i_name} vectoring on {role_symbol} {t_callsign} ({t_name})."))
            if is_airship:
                global_chatter.append((0.0, f"AWACS: Alert! Command Cruiser {t_callsign} is airborne and active!"))
                
            t_sim = 0.0
            while t_sim < max_time and t_sim <= death_time + 1.5 and not unit_withdrawn:
                # 1. Update Target (Evasions/Maneuvers/Withdrawals)
                if not target_destroyed:
                    # Check if target should withdraw
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
                        # Fly straight away from the base
                        th_rad = math.atan2(tx - bx, ty - by)
                        tv = eng['velocity'] * 1.3
                        tvz = -30.0 # descend slowly
                        # Check exit boundary
                        dist_b = math.sqrt((tx - bx)**2 + (ty - by)**2)
                        if dist_b > 110000.0:
                            unit_withdrawn = True
                            global_chatter.append((t_sim, f"AWACS: {t_callsign} has retreated from tactical radar range."))
                    elif incoming_missile and not is_airship:
                        # Helix climb or slalom weave maneuver
                        th_rad += math.sin(t_sim * 2.5) * 0.18 + 0.06
                        tvz = math.cos(t_sim * 2.0) * 80.0
                        
                        # Decoys deployment
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
                            th_rad += 0.005 # slow turn
                        else:
                            th_rad += 0.002
                            
                    # Update Target Velocity and Coordinates
                    tvx = tv * math.sin(th_rad)
                    tvy = tv * math.cos(th_rad)
                    tx += tvx * dt
                    ty += tvy * dt
                    tz = np.clip(tz + tvz * dt, 200.0, 20000.0)
                    
                # 2. Update Decoys
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
                
                # 3. Update Interceptor / Scramble flight path
                dist_ti = math.sqrt((tx - ix)**2 + (ty - iy)**2 + (tz - iz)**2)
                
                # Steer towards target or withdraw
                dx = tx - ix
                dy = ty - iy
                dz = tz - iz
                if dist_ti > 0 and (i_weapon in ['laser', 'railgun'] or not missiles) and not target_destroyed and target_status != 'withdrawing':
                    ix += (dx / dist_ti) * iv * dt
                    iy += (dy / dist_ti) * iv * dt
                    iz += (dz / dist_ti) * iv * dt
                else:
                    # Target is destroyed or withdrawing: interceptor withdraws to base
                    dist_to_base = math.sqrt((ix - bx)**2 + (iy - by)**2 + (iz - bz)**2)
                    if dist_to_base > 4000.0:
                        ix += ((bx - ix) / dist_to_base) * iv * dt
                        iy += ((by - iy) / dist_to_base) * iv * dt
                        iz += ((bz - iz) / dist_to_base) * iv * dt
                        if t_sim % 5.0 < 0.1 and not target_destroyed and interceptor_status == 'active':
                            interceptor_status = 'withdrawing'
                            global_chatter.append((t_sim, f"{i_name}: Target retreated/splashed. Withdrawing to base."))
                    else:
                        # Landed
                        pass
                        
                # Weapons Firing logic (only if target is active and not withdrawing)
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
                                    global_chatter.append((t_sim, f"AWACS: Railgun hit! Bogey {t_callsign} vaporized!"))
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
                        
                    # Giant Airship Counter-Battery
                    if is_airship and dist_ti < 25000.0 and t_sim % 6.0 < 0.2:
                        global_chatter.append((t_sim, f"{t_callsign}: Activating defense grid! Firing SAM battery!"))
                        global_chatter.append((t_sim, f"{i_name}: Threat alert! Missile on tail!"))
                        
                # 4. Update Missiles (PropNav Guidance)
                next_missiles = []
                for m in missiles:
                    # Check decoy locks
                    if m['target_locked'] == 'target':
                        for d_idx, d in enumerate(decoys):
                            dist_md = math.sqrt((m['x'] - d['x'])**2 + (m['y'] - d['y'])**2 + (m['z'] - d['z'])**2)
                            if dist_md < 5000.0:
                                if random.random() < (0.65 if d['type'] == 'flare' else 0.45):
                                    m['target_locked'] = d_idx
                                    global_chatter.append((t_sim, f"AWACS: Target deployed chaff/flares. Lock transferred!"))
                                    break
                                    
                    # Homing vector
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
                        
                        # Steer weight for turning rate constraints
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
                    
                    # Hit check
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
                
                # Log snapshot
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
            
        # Simulate 12 CAP fighters globally
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
                    # Calculate position on patrol orbit
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
                    
                    # Check proximity to active hostile bogeys
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
                                        
                    # Check if it's time to withdraw
                    if c['status'] == 'patrolling' and t_sim > c['withdraw_time']:
                        c['status'] = 'withdrawing'
                        c['withdraw_start'] = t_sim
                        bx, by = self.base_x[c['base_idx']], self.base_y[c['base_idx']]
                        angle = math.atan2(cx - bx, cy - by)
                        c['withdraw_vx'] = 550.0 * math.sin(angle)
                        c['withdraw_vy'] = 550.0 * math.cos(angle)
                        global_chatter.append((t_sim, f"{c['callsign']}: Bingo fuel. Withdrawing from patrol orbit."))
                        
                elif c['status'] == 'withdrawing':
                    # Update coordinate flying outward
                    c['x'] += c['withdraw_vx'] * dt
                    c['y'] += c['withdraw_vy'] * dt
                    c['z'] = max(100.0, c['z'] - 80.0 * dt)
                    
                    bx, by = self.base_x[c['base_idx']], self.base_y[c['base_idx']]
                    dist = math.sqrt((c['x'] - bx)**2 + (c['y'] - by)**2)
                    if dist > 105000.0:
                        c['status'] = 'withdrawn'
                        
                elif c['status'] == 'destroyed':
                    # stays at death position
                    pass
                    
                # Append to history
                c['history'].append({
                    'time': t_sim,
                    'tx': c['x'], 'ty': c['y'], 'tz': c['z'],
                    'status': c['status'],
                    'destroyed': (c['status'] == 'destroyed')
                })
            t_sim += dt
            
        return results, cap_fighters, global_chatter

# Run Simulation
simulator = EngagementSimulator(engagements, base_x_arr, base_y_arr, base_z_arr)
eng_results, cap_results, chatter_log = simulator.run_simulation(max_time=30.0, dt=0.25)
chatter_log.sort(key=lambda x: x[0])

# Initial positions setup for traces
hostile_x, hostile_y, hostile_z, hostile_txt, hostile_sym, hostile_size = [], [], [], [], [], []
allied_x, allied_y, allied_z, allied_txt, allied_sym, allied_size = [], [], [], [], [], []

for eng in eng_results:
    h0 = eng['history'][0]
    is_airship = (eng['aircraft_type_id'] == 18 or eng['aircraft_type_id'] == 19)
    sym = 'square' if is_airship else 'diamond'
    sz = 20 if is_airship else 8
    
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

# Add hostiles (Trace 7)
fig.add_trace(go.Scatter3d(
    x=hostile_x, y=hostile_y, z=hostile_z,
    mode='markers+text', marker=dict(size=hostile_size, color='red', symbol=hostile_sym),
    text=hostile_txt, textposition='top center',
    name='Hostile Fleet'
))

# Add Allied Scrambles (Trace 8)
fig.add_trace(go.Scatter3d(
    x=allied_x, y=allied_y, z=allied_z,
    mode='markers+text', marker=dict(size=allied_size, color='cyan', symbol=allied_sym),
    text=allied_txt, textposition='bottom center',
    name='Allied Scrambles'
))

# Add Hit explosions (Trace 9)
fig.add_trace(go.Scatter3d(
    x=[None], y=[None], z=[None],
    mode='markers', marker=dict(size=1, color='orange', symbol='circle'),
    name='Air Explosions'
))

# Allied CAP formations initial positions
patrol_x, patrol_y, patrol_z = [], [], []
for cap in cap_results:
    h0 = cap['history'][0]
    patrol_x.append(h0['tx'])
    patrol_y.append(h0['ty'])
    patrol_z.append(h0['tz'])

# Add CAP Patrol (Trace 10)
fig.add_trace(go.Scatter3d(
    x=patrol_x, y=patrol_y, z=patrol_z,
    mode='markers+text', marker=dict(size=7, color='rgba(0, 255, 200, 0.7)', symbol='diamond'),
    text=[c['callsign'] for c in cap_results],
    textposition='top center',
    name='CAP Patrols'
))

# Allied Trails (Trace 11)
fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='lines', line=dict(color='rgba(0, 255, 255, 0.55)', width=2), name='Allied Trails'))
# Enemy Trails (Trace 12)
fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='lines', line=dict(color='rgba(255, 80, 80, 0.55)', width=2), name='Enemy Trails'))
# Missile Trails (Trace 13)
fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='lines', line=dict(color='rgba(255, 255, 0, 0.8)', width=1.5), name='Missile Homing Lines'))
# Flares (Trace 14)
fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', marker=dict(size=4, color='rgba(255, 215, 0, 0.85)', symbol='circle'), name='Flares (Heat Decoy)'))
# Chaff (Trace 15)
fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', marker=dict(size=4, color='rgba(200, 200, 255, 0.85)', symbol='circle'), name='Chaff (Radar Decoy)'))
# Lasers (Trace 16)
fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='lines', line=dict(color='rgba(255, 0, 100, 0.85)', width=3), name='TLS Laser Beams'))
# Railgun spiral (Trace 17)
fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='lines', line=dict(color='rgba(150, 50, 255, 0.85)', width=1.5, dash='dash'), name='EML Railgun Electric Trails'))

# Lock-On Target Alert (Trace 18)
fig.add_trace(go.Scatter3d(
    x=[None], y=[None], z=[None],
    mode='lines',
    line=dict(color='rgba(255, 68, 68, 0.85)', width=2.0, dash='dash'),
    name='Lock-On Target Alert'
))

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

# Animation Frames
num_frames = 65
times = np.linspace(0.0, 30.0, num_frames)

# Calculate bounding box of active dogfights, bases, and CAP patrols to zoom in
xs_all = [0.0, -60000.0, 50000.0]
ys_all = [0.0, 40000.0, -50000.0]
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
    # Active nodes
    h_x, h_y, h_z, h_txt, h_sym, h_sz = [], [], [], [], [], []
    a_x, a_y, a_z, a_txt, a_sym, a_sz = [], [], [], [], [], []
    exp_x, exp_y, exp_z, exp_sz = [], [], [], []
    lock_x, lock_y, lock_z = [], [], []
    
    # Trails
    allied_trail_x, allied_trail_y, allied_trail_z = [], [], []
    enemy_trail_x, enemy_trail_y, enemy_trail_z = [], [], []
    missile_trail_x, missile_trail_y, missile_trail_z = [], [], []
    flare_x, flare_y, flare_z = [], [], []
    chaff_x, chaff_y, chaff_z = [], [], []
    laser_x, laser_y, laser_z = [], [], []
    railgun_x, railgun_y, railgun_z = [], [], []
    
    for eng in eng_results:
        # Get history state at closest time
        hist = eng['history']
        entry = min(hist, key=lambda x: abs(x['time'] - t))
        is_airship = (eng['aircraft_type_id'] == 18 or eng['aircraft_type_id'] == 19)
        sym = 'square' if is_airship else 'diamond'
        sz = 20 if is_airship else 8
        
        # Add trail paths up to time t
        e_path = [ (h['tx'], h['ty'], h['tz']) for h in hist if h['time'] <= entry['time'] ]
        i_path = [ (h['ix'], h['iy'], h['iz']) for h in hist if h['time'] <= entry['time'] ]
        
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
        
        # Current active plane marker
        if not entry['destroyed'] or entry['time'] >= t - 1.5:
            # Still visible or just exploded
            px_t, py_t, pz_t = entry['tx'], entry['ty'], entry['tz']
            ix_t, iy_t, iz_t = entry['ix'], entry['iy'], entry['iz']
            
            # Check lock status for target plane label styling
            is_locked_on = False
            for m in entry['missiles']:
                is_locked_on = True
            
            if not is_locked_on:
                dist_lock = math.sqrt((entry['tx'] - entry['ix'])**2 + (entry['ty'] - entry['iy'])**2 + (entry['tz'] - entry['iz'])**2)
                if dist_lock < 25000.0 and not entry['destroyed'] and entry['status'] != 'withdrawing':
                    is_locked_on = True
            
            label_txt = f"⚠️ [LOCK] {eng['callsign']}" if is_locked_on else eng['callsign']
            
            # Decouple IFF for styling
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
                
            # If hit exploded
            if entry['destroyed'] and entry['time'] <= t:
                exp_x.append(px_t)
                exp_y.append(py_t)
                exp_z.append(pz_t)
                exp_sz.append(int(12 + (t - entry['time']) * 14))
                
        # Missiles homing & lock-on lines
        missile_locked = False
        for m in entry['missiles']:
            for mx_p, my_p, mz_p in zip(m['px'], m['py'], m['pz']):
                missile_trail_x.append(mx_p)
                missile_trail_y.append(my_p)
                missile_trail_z.append(mz_p)
            missile_trail_x.append(None)
            missile_trail_y.append(None)
            missile_trail_z.append(None)
            
            # Lock-on line from missile to target
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
            
        # Flares
        for fx_p, fy_p, fz_p in entry['flares']:
            flare_x.append(fx_p)
            flare_y.append(fy_p)
            flare_z.append(fz_p)
            
        # Chaff
        for cx_p, cy_p, cz_p in entry['chaff']:
            chaff_x.append(cx_p)
            chaff_y.append(cy_p)
            chaff_z.append(cz_p)
            
        # Lasers
        for l in entry['lasers']:
            laser_x.extend([l['x0'], l['x1'], None])
            laser_y.extend([l['y0'], l['y1'], None])
            laser_z.extend([l['z0'], l['z1'], None])
            
        # Railguns
        for r_slug in entry['railguns']:
            rx_h, ry_h, rz_h = get_railgun_helix(r_slug['x0'], r_slug['y0'], r_slug['z0'], r_slug['x1'], r_slug['y1'], r_slug['z1'])
            railgun_x.extend(rx_h + [None])
            railgun_y.extend(ry_h + [None])
            railgun_z.extend(rz_h + [None])

    # Allied CAP Patrol orbits (updating with dynamic CAP results)
    pat_x, pat_y, pat_z, pat_txt = [], [], [], []
    for cap in cap_results:
        hist = cap['history']
        entry_c = min(hist, key=lambda x: abs(x['time'] - t))
        if entry_c['status'] != 'withdrawn':
            if entry_c['status'] != 'destroyed' or entry_c['time'] >= t - 1.5:
                pat_x.append(entry_c['tx'])
                pat_y.append(entry_c['ty'])
                pat_z.append(entry_c['tz'])
                
                # Check lock on CAP unit
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
                
                # If just destroyed, add to explosions list!
                if entry_c['status'] == 'destroyed' and entry_c['time'] <= t:
                    exp_x.append(entry_c['tx'])
                    exp_y.append(entry_c['ty'])
                    exp_z.append(entry_c['tz'])
                    exp_sz.append(int(12 + (t - entry_c['time']) * 14))

    # Sync radio chatter logs (take last 5 messages)
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

# UI layout
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

import plotly.io as pio
pio.renderers.default = "colab"
fig.show()
""")

# Cell 14
add_markdown("""## 5. Comparative Data Processing Sandbox (PySpark vs. CUDA)

To satisfy the data processing coursework options and demonstrate absolute control over processing paradigms, this section compares **four parallel and sequential execution strategies** on 1,000,000 radar records.

We calculate the **Kinetic Hazard Index ($K_i$)** for all 1,000,000 aircraft records:
$$K_i = \\begin{cases} \\frac{v_i^2}{2.0 \\cdot (d_i + 1.0)} & \\text{if Hostile } (iff\\_status == 0) \\\\ 0.0 & \\text{if Allied } (iff\\_status == 1) \\end{cases}$$
where $d_i = \\sqrt{x_i^2 + y_i^2 + \\text{altitude}_i^2}$ (distance to center base).

Then, we apply a **Filter** to keep only records where $K_i > 0.1$, and perform a **Reduce** (aggregation) to find:
1. The **Count** of active bogeys exceeding this hazard threshold.
2. The **Sum** of $K_i$ for all filtered bogeys.
3. The **Maximum** $K_i$ observed in the airspace.

We compare:
1. **PySpark SQL / DataFrame (Spark Session)**
2. **PySpark RDD Map/Filter/Reduce (Spark Context)**
3. **CUDA GPGPU with Shared Memory block reduction (Grids & Threads)**
4. **Single-threaded NumPy / Python CPU baseline**
""")

# Cell 15
add_code("""# Option A: PySpark SQL/DataFrames (Spark Session)
import time
from pyspark.sql.functions import col, sqrt, when, sin, cos, sum as spark_sum, max as spark_max
import math

print("Option A: Executing PySpark DataFrame / SQL (Spark Session) on 1,000,000 rows...")

# Caching & warm-up to ensure timing reflects JVM performance, not disk I/O or lazy compilation
df_cached = df.cache()
df_cached.count() # Force Spark to load and cache the dataset

start_time = time.time()

# 1. Map: Calculate Kinetic Hazard Index (K) using vector projection math
df_processed = df_cached.withColumn("dist", sqrt(col("x")**2 + col("y")**2 + col("altitude")**2)) \
                        .withColumn("vx", col("velocity") * sin(col("heading") * math.pi / 180.0)) \
                        .withColumn("vy", col("velocity") * cos(col("heading") * math.pi / 180.0)) \
                        .withColumn("dot", col("x") * col("vx") + col("y") * col("vy")) \
                        .withColumn("v_close", when(-col("dot") / (col("dist") + 1.0) > 0, -col("dot") / (col("dist") + 1.0)).otherwise(0.0)) \
                        .withColumn("K", when(col("iff_status") == 0, (col("v_close")**2) / (2.0 * (col("dist") + 1.0))).otherwise(0.0))

# 2. Filter: Find high-hazard bogeys (> 0.1)
df_filtered = df_processed.filter((col("iff_status") == 0) & (col("K") > 0.1))

# 3. Reduce: Aggregate count, sum, and max of K
results_df = df_filtered.select(
    spark_sum("K").alias("total_k"),
    spark_max("K").alias("max_k")
).first()

spark_count = df_filtered.count()
spark_sum_val = results_df["total_k"] if results_df["total_k"] is not None else 0.0
spark_max_val = results_df["max_k"] if results_df["max_k"] is not None else 0.0

spark_df_time = time.time() - start_time
print(f"Spark DF Count: {spark_count}")
print(f"Spark DF Sum  : {spark_sum_val:.4f}")
print(f"Spark DF Max  : {spark_max_val:.4f}")
print(f"Spark DF Execution Time (Warmed Up): {spark_df_time:.4f} seconds")
""")

# Cell 16
add_code("""# Option B: PySpark RDD Map/Filter/Reduce (Spark Context)
import math

print("Option B: Executing PySpark RDD Map-Filter-Reduce (Spark Context) on 1,000,000 rows...")
start_time = time.time()

# Extract RDD from cached DataFrame
rdd = df_cached.rdd

# 1. Map: Process each row to compute K using vector projection
def map_row(row):
    px, py, pz = row["x"], row["y"], row["altitude"]
    v = row["velocity"]
    h = row["heading"]
    iff = row["iff_status"]
    
    h_rad = h * math.pi / 180.0
    vx = v * math.sin(h_rad)
    vy = v * math.cos(h_rad)
    
    dist = math.sqrt(px**2 + py**2 + pz**2)
    dot = px*vx + py*vy
    v_close = -dot / (dist + 1.0)
    if v_close < 0.0:
        v_close = 0.0
        
    k = 0.0
    if iff == 0:
        k = (v_close**2) / (2.0 * (dist + 1.0))
    return (iff, k)

rdd_mapped = rdd.map(map_row)

# 2. Filter: Retain only records passing the threshold (> 0.1)
rdd_filtered = rdd_mapped.filter(lambda x: x[0] == 0 and x[1] > 0.1)

# 3. Reduce: Aggregate sum and max, and count elements
rdd_count = rdd_filtered.count()
if rdd_count > 0:
    # Reduce returns (sum_k, max_k)
    rdd_sum_val, rdd_max_val = rdd_filtered.map(lambda x: (x[1], x[1])).reduce(
        lambda a, b: (a[0] + b[0], max(a[1], b[1]))
    )
else:
    rdd_sum_val, rdd_max_val = 0.0, 0.0

rdd_time = time.time() - start_time
print(f"Spark RDD Count: {rdd_count}")
print(f"Spark RDD Sum  : {rdd_sum_val:.4f}")
print(f"Spark RDD Max  : {rdd_max_val:.4f}")
print(f"Spark RDD Execution Time: {rdd_time:.4f} seconds")
""")

# Cell 17
add_code("""# Option C: CUDA Shared Memory (Grids & Threads)
import ctypes
import numpy as np
import os

# Load ctypes signature for process_radar_data
lib_path = './libtrajectory.so' if os.path.exists('./libtrajectory.so') else './trajectory.dll'
lib_proc = ctypes.CDLL(lib_path)

lib_proc.process_radar_data.argtypes = [
    ctypes.c_int,                                            # num_records
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),  # x, y, altitude
    ctypes.POINTER(ctypes.c_float),                          # velocity
    ctypes.POINTER(ctypes.c_float),                          # heading
    ctypes.POINTER(ctypes.c_int),                            # iff_status
    ctypes.c_float,                                          # threshold
    ctypes.POINTER(ctypes.c_float),                          # out_k (output)
    ctypes.POINTER(ctypes.c_int),                            # out_filtered_count
    ctypes.POINTER(ctypes.c_float),                          # out_sum_k
    ctypes.POINTER(ctypes.c_float)                           # out_max_k
]

# Fetch 1,000,000 records into local memory arrays (selecting heading)
all_data = df_cached.select("x", "y", "altitude", "velocity", "heading", "iff_status").toPandas()
num_records = len(all_data)

x_in = np.array(all_data['x'], dtype=np.float32)
y_in = np.array(all_data['y'], dtype=np.float32)
z_in = np.array(all_data['altitude'], dtype=np.float32)
v_in = np.array(all_data['velocity'], dtype=np.float32)
h_in = np.array(all_data['heading'], dtype=np.float32)
iff_in = np.array(all_data['iff_status'], dtype=np.int32)

# Output buffers
out_k = np.zeros(num_records, dtype=np.float32)
out_count = ctypes.c_int(0)
out_sum = ctypes.c_float(0.0)
out_max = ctypes.c_float(0.0)

print(f"Option C: Executing CUDA Shared Memory GPGPU Kernel (Grid + Thread) on {num_records} rows...")
start_time = time.time()

lib_proc.process_radar_data(
    num_records,
    x_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    y_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    z_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    v_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    h_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    iff_in.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    ctypes.c_float(0.1),
    out_k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    ctypes.byref(out_count),
    ctypes.byref(out_sum),
    ctypes.byref(out_max)
)

cuda_time = time.time() - start_time
print(f"CUDA Count: {out_count.value}")
print(f"CUDA Sum  : {out_sum.value:.4f}")
print(f"CUDA Max  : {out_max.value:.4f}")
print(f"CUDA Execution Time (including HtoD/DtoH transfer): {cuda_time:.4f} seconds")
""")

# Cell 18
add_code("""# Option D: NumPy CPU Sequential Baseline
print("Option D: Executing NumPy/Python CPU Sequential Baseline on 1,000,000 rows...")
start_time = time.time()

# 1. Map: Compute distance, velocity vectors and K
cpu_dists = np.sqrt(x_in**2 + y_in**2 + z_in**2)
cpu_vx = v_in * np.sin(h_in * np.pi / 180.0)
cpu_vy = v_in * np.cos(h_in * np.pi / 180.0)
cpu_dot = x_in * cpu_vx + y_in * cpu_vy
cpu_v_close = -cpu_dot / (cpu_dists + 1.0)
cpu_v_close = np.clip(cpu_v_close, 0.0, None)

cpu_k = np.zeros(num_records, dtype=np.float32)
hostile_mask = (iff_in == 0)
cpu_k[hostile_mask] = (cpu_v_close[hostile_mask]**2) / (2.0 * (cpu_dists[hostile_mask] + 1.0))

# 2. Filter
filter_mask = hostile_mask & (cpu_k > 0.1)

# 3. Reduce
cpu_count = np.sum(filter_mask)
cpu_sum_val = np.sum(cpu_k[filter_mask]) if cpu_count > 0 else 0.0
cpu_max_val = np.max(cpu_k[filter_mask]) if cpu_count > 0 else 0.0

cpu_time = time.time() - start_time
print(f"CPU Baseline Count: {cpu_count}")
print(f"CPU Baseline Sum  : {cpu_sum_val:.4f}")
print(f"CPU Baseline Max  : {cpu_max_val:.4f}")
print(f"CPU Baseline Execution Time: {cpu_time:.4f} seconds")
""")

# Cell 19
add_code("""# Analytical Validation
print("===== Verification & Correctness Report =====")
print(f"Count Matches? {spark_count == rdd_count == out_count.value == cpu_count} ({spark_count} vs {rdd_count} vs {out_count.value} vs {cpu_count})")
sum_diff = abs(spark_sum_val - out_sum.value)
print(f"Sum matches? (Within numerical tolerance): {sum_diff < 1.0} (Spark: {spark_sum_val:.2f}, CUDA: {out_sum.value:.2f}, Diff: {sum_diff:.4f})")
max_diff = abs(spark_max_val - out_max.value)
print(f"Max matches? (Within numerical tolerance): {max_diff < 0.01} (Spark: {spark_max_val:.4f}, CUDA: {out_max.value:.4f})")
print("=============================================")
""")

# Cell 20
add_code("""# Performance Benchmark Visualization (Multiple Timing Graphs)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

labels = ['PySpark SQL (Session)', 'PySpark RDD (Context)', 'CUDA Shared Mem (GPU)', 'NumPy CPU (Sequential)']
times = [spark_df_time, rdd_time, cuda_time, cpu_time]
throughputs = [num_records / t / 1e6 for t in times] # Million records/sec

# Create subplots for the three views: Linear Time, Log Time, and Throughput
fig_perf = make_subplots(
    rows=1, cols=3,
    subplot_titles=(
        "Linear Execution Time (s)<br><sup>Lower is better</sup>",
        "Log Execution Time (s)<br><sup>Lower is better (Log Scale)</sup>",
        "Data Processing Throughput<br><sup>Higher is better (M-Recs/sec)</sup>"
    ),
    horizontal_spacing=0.1
)

# 1. Linear Time Bar Chart
fig_perf.add_trace(
    go.Bar(
        x=labels, y=times,
        marker=dict(color=times, coloraxis="coloraxis"),
        text=[f"{t:.4f}s" for t in times],
        textposition='auto',
        name="Linear Time"
    ),
    row=1, col=1
)

# 2. Log Time Bar Chart
fig_perf.add_trace(
    go.Bar(
        x=labels, y=times,
        marker=dict(color=times, coloraxis="coloraxis"),
        text=[f"{t:.4f}s" for t in times],
        textposition='auto',
        name="Log Time"
    ),
    row=1, col=2
)

# 3. Throughput Bar Chart
fig_perf.add_trace(
    go.Bar(
        x=labels, y=throughputs,
        marker=dict(color=throughputs, coloraxis="coloraxis2"),
        text=[f"{tp:.2f}M" for tp in throughputs],
        textposition='auto',
        name="Throughput"
    ),
    row=1, col=3
)

# Configure axes
fig_perf.update_yaxes(type="log", row=1, col=2, title_text="Seconds (Log Scale)")
fig_perf.update_yaxes(title_text="Seconds", row=1, col=1)
fig_perf.update_yaxes(title_text="Million Records / Sec", row=1, col=3)

fig_perf.update_layout(
    title_text="Big Data Processing Performance Battleground: PySpark vs. CUDA Shared Memory",
    template="plotly_dark",
    height=550,
    width=1100,
    showlegend=False,
    coloraxis=dict(colorscale="Viridis", showscale=False),
    coloraxis2=dict(colorscale="Cividis", showscale=False),
    margin=dict(t=100, b=50, l=50, r=50)
)

fig_perf.show()
""")

# Cell 21
add_markdown("""### 🧠 Deep Dive: Why are the timings so different?

#### 1. **CUDA GPGPU with Shared Memory (GPU)** ➔ 🥇 **1st Place (Ultra-Fast)**
- **Under the Hood**: Massively parallel execution across thousands of cores with block-level caching.
- **Why it wins**: 
  - Computes the vector math concurrently inside GPU registers.
  - Uses L1-speed **Shared Memory (`__shared__`)** to accumulate thread-level hazard counts, sums, and maximum values.
  - Reduces global VRAM atomic congestion by a factor of 256, finishing the 1,000,000 row calculation in a fraction of a millisecond.

#### 2. **NumPy CPU (Sequential C-Vectorized)** ➔ 🥈 **2nd Place (Fast)**
- **Under the Hood**: Standard single-threaded C-loops operating on contiguous memory arrays.
- **Why it beats Spark locally**: 
  - Running in-process inside the Python runtime means **zero inter-process communication (IPC)** and **zero scheduling overhead**.
  - For 1,000,000 records, NumPy's C-compiled vectorization crunches arrays directly in CPU Cache, bypassing JVM coordination.

#### 3. **PySpark SQL / DataFrame (Spark Session)** ➔ 🥉 **3rd Place (Moderate)**
- **Under the Hood**: Uses Spark's **Catalyst Optimizer** to build query plans and **Tungsten** for off-heap binary format data layout.
- **The Local Bottleneck**: 
  - Even though it uses JIT code generation, Spark is a distributed cluster engine. Running locally means it incurs JVM boot, partition slicing, task compilation, and thread-scheduling overhead.
  - On 1,000,000 rows, this coordination overhead is far larger than the actual computation time, making it slower than raw NumPy CPU.

#### 4. **PySpark RDD Map/Filter/Reduce (Spark Context)** ➔ 💀 **4th Place (Slowest)**
- **Under the Hood**: Row-by-row iteration using custom Python function maps.
- **Why it is so slow**: 
  - Spark's core is Java/Scala (JVM), but RDD functions must execute in Python workers.
  - Every single row must be serialized (using Py4J / Pickle), sent across local sockets from the JVM to the Python process, computed, and serialized back to the JVM. This **serialization bottleneck** creates severe CPU idle cycles.
""")

with open('colab_notebook.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("colab_notebook.ipynb created successfully.")
