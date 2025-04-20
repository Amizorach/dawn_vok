import random

import uuid

sensor_types = [
    "temperature", "humidity", "radiation", "soil_moisture", "air_pressure",
    "wind_speed", "co2", "ph", "ec", "light_intensity", "rainfall", "wind_direction", "wind_speed", "solar_radiation", "soil_temperature", "soil_moisture", "soil_ph", "soil_ec", "soil_organic_matter", "soil_nitrogen", "soil_phosphorus", "soil_potassium"
]

manufacturers = ["Netafim", "Bosch", "Parrot", "AgriTech", "SensorCo", "WeatherCo", "PlantCo", "SoilCo", "WaterCo", "LightCo"]
locations = ["Greenhouse A", "Greenhouse B", "Field C", "Field D", "Orchard E", "Farm F", "Field G", "Field H", "Field I", "Field J"]
models = [f"M{100 + i}" for i in range(20)]  # e.g., M100–M119

UNIT_DEFINITIONS = {
    str(uuid.uuid4()): {
        "unit_type": "sensor",
        "parameter": random.choice(sensor_types),
        "descriptors": {
            "manufacturer": random.choice(manufacturers),
            "location": random.choice(locations),
            "model": random.choice(models)
        }
    }
    for _ in range(1000)
}