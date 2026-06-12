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

# Override standard Temp directory to avoid Windows PermissionError on Java gateway file reads
local_temp = os.path.abspath("./tmp")
os.makedirs(local_temp, exist_ok=True)
os.environ['TEMP'] = local_temp
os.environ['TMP'] = local_temp

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
    .config("spark.local.dir", os.path.join(local_temp, "spark-local")) \
    .getOrCreate()

print("Spark Session created successfully.")
""")

# Cell 6
add_code("""# Load Data and Dynamic Base Coordinates
import json
import os

if os.path.exists("bases_config.json"):
    with open("bases_config.json", "r") as f:
        bases_data = json.load(f)
    bases_list = bases_data["bases"]
else:
    # Fallback default values
    bases_list = [
        {"name": "Alpha (HQ)", "x": 0.0, "y": 0.0, "z": 0.0},
        {"name": "Bravo (FOB)", "x": -60000.0, "y": 40000.0, "z": 0.0},
        {"name": "Charlie (Naval)", "x": 50000.0, "y": -50000.0, "z": 0.0}
    ]

bx_alpha, by_alpha = bases_list[0]["x"], bases_list[0]["y"]
bx_bravo, by_bravo = bases_list[1]["x"], bases_list[1]["y"]
bx_charlie, by_charlie = bases_list[2]["x"], bases_list[2]["y"]

df = spark.read.csv("radar_data.csv", header=True, inferSchema=True)

from pyspark.sql.functions import col, sqrt, atan2, degrees, abs as pyspark_abs, when, least, cos, lit
import math

# 2. Feature Engineering: Calculate distances to each of our 3 bases in 2D (xy plane)
df = df.withColumn("dist_alpha", sqrt((col("x") - bx_alpha)**2 + (col("y") - by_alpha)**2))
df = df.withColumn("dist_bravo", sqrt((col("x") - bx_bravo)**2 + (col("y") - by_bravo)**2))
df = df.withColumn("dist_charlie", sqrt((col("x") - bx_charlie)**2 + (col("y") - by_charlie)**2))

# 3. Feature Engineering: Find minimum distance to the closest base
df = df.withColumn("distance", least("dist_alpha", "dist_bravo", "dist_charlie"))

# 4. Feature Engineering: Get the coordinates of the closest base
df = df.withColumn("bx_closest", 
    when(col("dist_alpha") <= col("dist_bravo"), 
        when(col("dist_alpha") <= col("dist_charlie"), bx_alpha).otherwise(bx_charlie)
    ).otherwise(
        when(col("dist_bravo") <= col("dist_charlie"), bx_bravo).otherwise(bx_charlie)
    )
)
df = df.withColumn("by_closest", 
    when(col("dist_alpha") <= col("dist_bravo"), 
        when(col("dist_alpha") <= col("dist_charlie"), by_alpha).otherwise(by_charlie)
    ).otherwise(
        when(col("dist_bravo") <= col("dist_charlie"), by_bravo).otherwise(by_charlie)
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

# Cell 7
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

# Cell 8
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

# Cell 9
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

# Define Base locations dynamically
import json
import os
if os.path.exists("bases_config.json"):
    with open("bases_config.json", "r") as f:
        bases_data = json.load(f)
    bases_list = bases_data["bases"]
else:
    bases_list = [
        {"name": "Alpha (HQ)", "x": 0.0, "y": 0.0, "z": 0.0},
        {"name": "Bravo (FOB)", "x": -60000.0, "y": 40000.0, "z": 0.0},
        {"name": "Charlie (Naval)", "x": 50000.0, "y": -50000.0, "z": 0.0}
    ]

base_x_arr = np.array([b["x"] for b in bases_list], dtype=np.float32)
base_y_arr = np.array([b["y"] for b in bases_list], dtype=np.float32)
base_z_arr = np.array([b["z"] for b in bases_list], dtype=np.float32)
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
    19: "X-02S STRIKE WYVERN (EML Railgun)",
    20: "X-02S STRIKE WYVERN (EML Railgun)",
    21: "X-02S STRIKE WYVERN (EML Railgun)",
    22: "X-02S STRIKE WYVERN (EML Railgun)",
    23: "X-02S STRIKE WYVERN (EML Railgun)",
    24: "Anura-class (Cruise Intercept)",
    25: "ADF-01 FALKEN (TLS Laser)",
    26: "SOLG Orbital Laser (TLS)"
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
        diffs = [abs(bx - b["x"]) for b in bases_list]
        closest_b_idx = diffs.index(min(diffs))
        b_name = bases_list[closest_b_idx]["name"]
            
    clock = get_clock_position(row['x'], row['y'], bx, by)
    elevation = get_elevation(row['altitude'])
    
    callsign_str = f"{row['callsign']} ({row['aircraft_type']})" if row['squadron_name'] != 'None' else row['aircraft_type']
    
    if row['tti'] > 0:
        matchup = matchup_dict.get(row['aircraft_type_id'], "SAM MISSILE")
        print(f"AWACS: \\\"Bandit {callsign_str}, hot at {clock} o'clock, {elevation}! Base {b_name} scrambling {matchup} interceptor. TTI {row['tti']:.1f}s!\\\"")
    else:
        print(f"AWACS: \\\"WARNING! High-speed Bandit {callsign_str}, {clock} o'clock, {elevation} relative to Base {b_name} has breached intercept envelope!\\\"")
print("================================\\n")
""")

# Cell 12
add_code("""# Interactive 3D Animated Tactical Map with Plotly
from scripts.visualize_airspace import show_tactical_map
show_tactical_map(top_threats, base_x_arr, base_y_arr, base_z_arr, predictions)
""")

# Cell 13
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

# Cell 14
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

# Cell 15
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

# Cell 16
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

# Cell 17
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

# Cell 18
add_code("""# Analytical Validation
print("===== Verification & Correctness Report =====")
print(f"Count Matches? {spark_count == rdd_count == out_count.value == cpu_count} ({spark_count} vs {rdd_count} vs {out_count.value} vs {cpu_count})")
sum_diff = abs(spark_sum_val - out_sum.value)
print(f"Sum matches? (Within numerical tolerance): {sum_diff < 1.0} (Spark: {spark_sum_val:.2f}, CUDA: {out_sum.value:.2f}, Diff: {sum_diff:.4f})")
max_diff = abs(spark_max_val - out_max.value)
print(f"Max matches? (Within numerical tolerance): {max_diff < 0.01} (Spark: {spark_max_val:.4f}, CUDA: {out_max.value:.4f})")
print("=============================================")
""")

# Cell 19
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

# Cell 20
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

with open('windows_notebook.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("windows_notebook.ipynb created successfully.")
