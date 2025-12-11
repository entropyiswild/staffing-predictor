import streamlit as st 		# makes the webpage
import pandas as pd		# reads csv + tables
import numpy as np		# for log values
import joblib			# opens pkl files

st.set_page_config(page_title="Live Events Staffing Predictor", layout="centered")
# ↑ sets browser tab title and centers for phones

st.title("Global Live Events Staffing & Cost Predictor")
st.markdown("Fill in the form → get exact hours + USD costs instantly")
# ↑ headline and instructions; what users see first

# ──────────────────────────────────────────────────────
# Load everything once; increases speed
# ──────────────────────────────────────────────────────
@st.cache_resource					# Streamlit setup magic
def load_all():
	models = joblib.load("models/my_staffing_models.pkl")	# opens trained brains/models
	features = joblib.load("models/my_feature_columns.pkl")	# opens list of expected columns
	venues = pd.read_csv("Venue Database.csv")		# venue info (capacity, entrances, ect.)
	rates = pd.read_csv("Local Rates Database.csv")		# local hourly wages
	return models, features, venues, rates

models, feature_cols, venue_df, rates_df = load_all()
# ↑ all is loaded into memory + ready

# ──────────────────────────────────────────────────────
# Input form (what is asked in planning meetings)
# ──────────────────────────────────────────────────────
col1, col2 = st.columns(2) 	# 2 columns for phones
with col1:
	venue_name = st.selectbox("Venue", venue_df["venue_name"])
	event_type = st.selectbox("Event Type", ["Concert","Sports Game","Theater"])
	predicted_attendance = st.number_input("Predicted Attendance", 1000, 100000, 18000)
	duration = st.slider("Duration (hours)",1.0 ,8.0 ,3.5)
with col2:
	peak_arrival = st.slider("Peak Arrival Window (minutes)", 30, 180, 90)
	sellout = st.slider("Sellout Likelihood", 0.0, 1.0, 0.85, 0.05)
	alcohol = st.checkbox("Alcohol Served", True)
	day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
	season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])

# ──────────────────────────────────────────────────────
# Big green button
# ──────────────────────────────────────────────────────
if st.button("Calculate Staffing & Costs"):
	# grab venue details
	venue_row = venue_df[venue_df["venue_name"] == venue_name].iloc[0]
	city = venue_row["city_country"]
	capacity = venue_row["capacity"]
	entrances = venue_row["num_entrances"]

	# Build exact input row used in training script/program
	data = {
		"duration_hours": duration,
		"log_predicted_attendance": np.log1p(predicted_attendance),
		"sellout_likelihood": sellout,
		"peak_arrival_minutes": peak_arrival,
		"alcohol_served": int(alcohol),
		"log_capacity": np.log1p(capacity),
		"num_entrances": entrances,
		"num_concessions": venue_row["num_concessions"],
		"predicted_density": predicted_attendance / capacity,
		"peak_per_entrance": peak_arrival / entrances,
		"is_weekend": 1 if day in ["Saturday", "Sunday"] else 0,
		"event_type": event_type,
		"day_of_week": day,
		"season": season
	}

	# turn data variable into a tiny one-row table
	df = pd.DataFrame([data])

	# one-hot encode exactly like training
	df = pd.get_dummies(df, columns=["event_type","day_of_week","season"])

	# add venue column as done in training
	df["venue_name_" + venue_name.replace(" ", "_")] = 1

	# ensure every column the model expects exists (fill missing with 0)
	for col in feature_cols:
		if col not in df.columns:
			df[col] = 0
	df = df[feature_cols]   		# put columns in exact order the brain/model expects

	# ask the models for their predictions and scale to real attendance
	scale = predicted_attendance / 1000
	preds = {pos: models[pos].predict(df)[0] * scale for pos in models}
	
	# look up hourly rates and calculate costs
	rate_row = rates_df[rates_df["city_country"] == city]
	costs = {}
	for pos in models:
		rate = rate_row[rate_row["position"] == pos.capitalize()].iloc[0]["hourly_rate_usd"]
		costs[pos] = round(preds[pos] * rate)
		
	total_h = sum(preds.values())
	total_c = sum(costs.values())

	# show the results table
	result = pd.DataFrame({
		"Position": ["Security","Concessions","Cleaning","Ushers","TOTAL"],
		"Hours": [round(preds[p]) for p in models] + [round(total_h)],
		"Cost (USD)": [costs[p] for p in models] + [total_c]
	})
	st.success("Prediction complete!")
	st.table(result)