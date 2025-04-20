import requests
import pandas as pd
import pprint
# === SETUP ===
API_TOKEN = "69bc0f2259614733b8e7e8a8cc5f8c3e"  # replace with your actual token
BASE_URL = "https://api.synopticdata.com/v2"

# === 1. Get all active RAWS stations in California ===
metadata_url = f"{BASE_URL}/stations/metadata"
# Bounding box around Sierra Nevada area (adjust as needed)
params_meta = {
    "token": API_TOKEN,
    "bbox": "-122.5,36.5,-118,39.5",  # (minLon, minLat, maxLon, maxLat)
    "status": "active",
    "within": 60,
    "format": "json"
}


# response = requests.get(metadata_url, params=params_meta)
# pprint.pp(response.json())
# stations = response.json().get('STATION', [])

# # Just get station IDs and coordinates
# station_info = [{
#     'stid': s['STID'],
#     'name': s['NAME'],
#     'lat': s['LATITUDE'],
#     'lon': s['LONGITUDE']
# } for s in stations]

# print(f"Found {len(station_info)} RAWS stations in California.")
# print(station_info[0])
# # === 2. Get real-time data for all these stations ===
# stids = ','.join([s['stid'] for s in station_info])
STATION_ID = 'TRMC1'                 # Replace with any valid STID
START_DATE = "202502010000"        # Format: YYYYMMDDHHMM
END_DATE = "202503020000"

# --- Set up API request ---
url = "https://api.synopticdata.com/v2/stations/timeseries"
params = {
    "token": API_TOKEN,
    "stid": STATION_ID,
    "start": START_DATE,
    "end": END_DATE,
    "vars": "air_temp,relative_humidity,wind_speed",
    "obtimezone": "local",     # or 'utc'
    "format": "json"
}

# --- Make the request ---
response = requests.get(url, params=params)
data = response.json()
pprint.pp(data)
# --- Parse the data ---
station_data = data["STATION"][0]["OBSERVATIONS"]

# Convert to DataFrame
df = pd.DataFrame({
    "time": station_data["date_time"],
    "temp_C": station_data.get("air_temp_set_1"),
    "rh": station_data.get("relative_humidity_set_1"),
    "wind_mps": station_data.get("wind_speed_set_1")
})

# --- Save to CSV ---
df.to_csv(f"{STATION_ID}_historical.csv", index=False)
print(f"Saved historical data for {STATION_ID} to CSV.")