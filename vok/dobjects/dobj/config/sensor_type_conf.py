class SensorTypeConfig:
    type_info = [
    {
        "uid": "air_temperature",
        "base_sensor_type": "temperature",
        "sensor_type": "temperature",
        "value_type": "float",
        "unit": "celsius",
        "norm_range": [-5.0, 50.0],
        "physical_quantity": "air_heat_content",
        "syntax_directives": [
            "Air temperature measures the thermal energy of the surrounding environment.",
            "It outputs values in degrees Celsius, normalized between -50 to 60.",
            "It continuously monitors diurnal temperature variations, capturing the daily cycle.",
            "Seasonal shifts in temperature are tracked to analyze yearly climate trends.",
            "Sensor data supports adaptive strategies in precision agriculture, ensuring optimal plant growth and resource management."
        ]
    },
    {
        "uid": "relative_humidity",
        "base_sensor_type": "humidity",
        "sensor_type": "humidity",
        "value_type": "float",
        "unit": "percentage",
        "norm_range": [0.0, 100.0],
        "physical_quantity": "water_vapor_ratio",
        "syntax_directives": [
            "Relative humidity represents the percentage of water vapor present in the air.",
            "It outputs values as percentages, normalized between 0 to 100.",
            "The sensor provides crucial insights into atmospheric moisture for both weather forecasting and climate analysis.",
            "It monitors daily fluctuations to help understand evaporation and condensation cycles impacting plant transpiration.",
            "Seasonal variations in humidity levels support irrigation planning and crop water stress management."
        ]
    },
    {
        "uid": "radiation",
        "base_sensor_type": "radiation",
        "sensor_type": "radiation",
        "value_type": "float",
        "unit": "W/m²",
        "norm_range": [0.0, 1400.0],
        "physical_quantity": "solar_energy_intensity",
        "syntax_directives": [
            "Radiation measures the amount of solar energy received per square meter.",
            "It is crucial for understanding the energy balance in agricultural and environmental systems.",
            "This sensor plays a vital role in monitoring plant photosynthetic activity.",
            "Values are reported in watts per square meter (W/m²) and influence crop growth analysis.",
            "Data helps in assessing both diurnal and seasonal variations in solar intensity."
        ]
    },
    {
        "uid": "soil_moisture",
        "base_sensor_type": "moisture",
        "sensor_type": "soil_moisture",
        "value_type": "float",
        "unit": "m³/m³",
        "norm_range": [0.0, 0.6],
        "physical_quantity": "volumetric_water_content",
        "syntax_directives": [
            "Soil moisture indicates the volumetric water content within the soil.",
            "It is essential for effective irrigation and plant water stress management.",
            "Accurate measurements support soil water balance and root zone analysis.",
            "Data is measured in cubic meters of water per cubic meter of soil (m³/m³).",
            "It facilitates real-time decisions in precision agriculture and resource optimization."
        ]
    },
    {
        "uid": "air_pressure",
        "base_sensor_type": "pressure",
        "sensor_type": "air_pressure",
        "value_type": "float",
        "unit": "hPa",
        "norm_range": [800.0, 1100.0],
        "physical_quantity": "atmospheric_pressure",
        "syntax_directives": [
            "Air pressure represents the atmospheric force exerted per unit area.",
            "It is a key parameter for weather prediction and environmental monitoring.",
            "The sensor assists in altitude adjustment and meteorological studies.",
            "Measurements are recorded in hectopascals (hPa).",
            "Understanding pressure variations aids in forecasting storm formation and climate shifts."
        ]
    },
    {
        "uid": "diffused_radiation",
        "base_sensor_type": "radiation",
        "sensor_type": "diffused_radiation",
        "value_type": "float",
        "unit": "W/m²",
        "norm_range": [0.0, 800.0],
        "physical_quantity": "scattered_solar_energy",
        "syntax_directives": [
            "Diffused radiation quantifies the scattered component of solar energy after atmospheric interaction.",
            "It is important for evaluating light conditions in shaded environments.",
            "The sensor aids in accurately modeling canopy light distribution and photosynthesis.",
            "Measurements are provided in watts per square meter (W/m²).",
            "This data is critical for optimizing crop layouts and shade management strategies."
        ]
    },
    {
        "uid": "global_radiation",
        "base_sensor_type": "radiation",
        "sensor_type": "global_radiation",
        "value_type": "float",
        "unit": "W/m²",
        "norm_range": [0.0, 1400.0],
        "physical_quantity": "total_solar_energy",
        "syntax_directives": [
            "Global radiation captures the complete solar energy incident on a horizontal surface.",
            "It encompasses both direct sunlight and diffused sky radiation.",
            "This comprehensive measurement is fundamental for energy balance studies.",
            "Data is expressed in watts per square meter (W/m²).",
            "It supports assessment of overall solar input for crop production and climatology."
        ]
    },
    {
        "uid": "max_temperature",
        "base_sensor_type": "temperature",
        "sensor_type": "max_temperature",
        "value_type": "float",
        "unit": "Celsius", # Assuming Celsius based on context, original code had 'Celsius'
        "norm_range": [-10.0, 60.0],
        "physical_quantity": "daily_max_temperature",
        "syntax_directives": [
            "Max temperature logs the highest recorded air temperature within a specified period.",
            "It is a crucial metric for detecting heat stress in plants and humans.",
            "The sensor output aids in identifying extreme weather conditions.",
            "Data is expressed in degrees Celsius for clarity in temperature range.",
            "It supports both short-term and long-term climate trend analysis."
        ]
    },
    {
        "uid": "wind_speed",
        "base_sensor_type": "wind",
        "sensor_type": "wind_speed",
        "value_type": "float",
        "unit": "m/s",
        "norm_range": [0.0, 60.0],
        "physical_quantity": "horizontal_wind_velocity",
        "syntax_directives": [
            "Wind speed quantifies the horizontal velocity of moving air.",
            "It is essential for understanding weather dynamics and dispersion phenomena.",
            "Measurements are critical for calculating evapotranspiration in agricultural models.",
            "Sensor readings are provided in meters per second (m/s).",
            "This data contributes to wind energy assessments and environmental modeling."
        ]
    },
    {
        "uid": "co2",
        "base_sensor_type": "gas",
        "sensor_type": "co2",
        "value_type": "float",
        "unit": "ppm",
        "norm_range": [300.0, 2000.0],
        "physical_quantity": "carbon_dioxide_concentration",
        "syntax_directives": [ # Using first definition encountered
            "CO2 sensor measures the concentration of carbon dioxide in the air.",
            "Relevant for photosynthesis modeling and greenhouse ventilation.",
            "Reported in parts per million (ppm)."
        ]
    },
    {
        "uid": "ph",
        "base_sensor_type": "chemical",
        "sensor_type": "ph",
        "value_type": "float",
        "unit": "pH",
        "norm_range": [3.5, 9.0],
        "physical_quantity": "hydrogen_ion_activity",
        "syntax_directives": [ # Using first definition encountered
            "pH represents the acidity or alkalinity of a substance.",
            "Essential for soil chemistry, nutrient uptake, and hydroponics.",
            "Values are unitless on a logarithmic scale."
        ]
    },
    {
        "uid": "light_intensity",
        "base_sensor_type": "radiation",
        "sensor_type": "light_intensity",
        "value_type": "float",
        "unit": "lux",
        "norm_range": [0, 120000],
        "physical_quantity": "visible_light_flux",
        "syntax_directives": [ # Using first definition encountered
            "Light intensity measures the amount of visible light reaching a surface.",
            "Important for plant photoperiodism, shading analysis, and indoor lighting.",
            "Reported in lux (lumens per square meter)."
        ]
    },
    {
        "uid": "soil_temperature",
        "base_sensor_type": "temperature",
        "sensor_type": "soil_temperature",
        "value_type": "float",
        "unit": "celsius",
        "norm_range": [-5.0, 45.0],
        "physical_quantity": "soil_temperature",
        "syntax_directives": [
            "Soil temperature sensor measures the thermal state of the soil profile.",
            "It is vital for understanding root zone dynamics and seed germination.",
            "The sensor captures both daily and seasonal temperature fluctuations.",
            "Accurate soil temperature data helps in optimizing irrigation and planting schedules.",
            "Monitoring soil temperature is key to assessing soil health and crop performance."
        ]
    },
    {
        "uid": "soil_ec",
        "base_sensor_type": "electrical",
        "sensor_type": "soil_ec",
        "value_type": "float",
        "unit": "dS/m",
        "norm_range": [0.0, 5.0],
        "physical_quantity": "soil_electrical_conductivity",
        "syntax_directives": [
            "Soil EC sensor measures the electrical conductivity of the soil solution.",
            "It provides insights into the soil’s salinity and ion concentration.",
            "High EC values may indicate excess salt, which can affect plant growth.",
            "The sensor aids in managing irrigation and fertilizer application strategies.",
            "Monitoring EC helps ensure sustainable soil management and optimal crop yield."
        ]
    },
    {
        "uid": "soil_organic_matter",
        "base_sensor_type": "chemical",
        "sensor_type": "soil_organic_matter",
        "value_type": "float",
        "unit": "percentage",
        "norm_range": [0.0, 20.0],
        "physical_quantity": "soil_organic_matter",
        "syntax_directives": [
            "This sensor quantifies the organic matter content in the soil.",
            "Organic matter is essential for improving soil fertility and water retention.",
            "It plays a crucial role in nutrient cycling and soil structure.",
            "Accurate readings support sustainable land management and organic amendment practices.",
            "Data from the sensor helps track soil quality and long-term ecological health."
        ]
    },
    {
        "uid": "soil_nitrogen",
        "base_sensor_type": "chemical",
        "sensor_type": "soil_nitrogen",
        "value_type": "float",
        "unit": "mg/kg",
        "norm_range": [0.0, 1000.0],
        "physical_quantity": "soil_nitrogen_content",
        "syntax_directives": [
            "Soil nitrogen sensor measures the available nitrogen content in the soil.",
            "Nitrogen is a key nutrient required for plant growth and protein synthesis.",
            "The sensor output informs fertilizer management and crop nutrition planning.",
            "It aids in evaluating soil fertility and guiding sustainable agricultural practices.",
            "Regular monitoring supports efficient nutrient use and improved crop yields."
        ]
    },
    {
        "uid": "soil_phosphorus",
        "base_sensor_type": "chemical",
        "sensor_type": "soil_phosphorus",
        "value_type": "float",
        "unit": "mg/kg",
        "norm_range": [0.0, 500.0],
        "physical_quantity": "soil_phosphorus_content",
        "syntax_directives": [
            "Soil phosphorus sensor assesses the phosphorus concentration available to plants.",
            "Phosphorus is crucial for energy transfer and root development in crops.",
            "Sensor data helps determine soil fertility and phosphorus deficiency.",
            "It supports precise fertilizer application to optimize plant growth.",
            "Monitoring phosphorus levels assists in maintaining balanced soil nutrient profiles."
        ]
    },
    {
        "uid": "soil_potassium",
        "base_sensor_type": "chemical",
        "sensor_type": "soil_potassium",
        "value_type": "float",
        "unit": "mg/kg",
        "norm_range": [0.0, 1500.0],
        "physical_quantity": "soil_potassium_content",
        "syntax_directives": [
            "Soil potassium sensor measures the concentration of potassium in the soil.",
            "Potassium is essential for water regulation and enzyme activation in plants.",
            "The sensor aids in assessing soil fertility and plant nutrient balance.",
            "It informs fertilizer strategies and helps improve crop stress resistance.",
            "Regular monitoring supports optimal nutrient management and sustainable yields."
        ]
    },
    {
        "uid": "dew_point",
        "base_sensor_type": "temperature",
        "sensor_type": "dew_point",
        "value_type": "float",
        "unit": "celsius",
        "norm_range": [-10.0, 30.0],
        "physical_quantity": "dew_point_temperature",
        "syntax_directives": [
            "Dew point sensor measures the temperature at which air becomes saturated.",
            "It indicates the onset of condensation and is critical for forecasting fog and dew formation.",
            "Measurements are reported in degrees Celsius.",
            "Data supports precise prediction of microclimatic conditions in agricultural and environmental studies.",
            "Accurate dew point readings are essential for scheduling irrigation and frost protection."
        ]
    },
    {
        "uid": "vp",
        "base_sensor_type": "gas",
        "sensor_type": "vp",
        "value_type": "float",
        "unit": "kPa",
        "norm_range": [0.0, 4.0],
        "physical_quantity": "vapor_pressure",
        "syntax_directives": [
            "Vapor pressure sensor measures the partial pressure exerted by water vapor in the air.",
            "It is vital for characterizing the moisture content in the atmosphere.",
            "Values are expressed in kilopascals (kPa).",
            "The sensor supports assessments of evaporative demand and plant transpiration.",
            "Accurate vapor pressure data enhances climate modeling and microclimate analysis."
        ]
    },
    {
        "uid": "vpd",
        "base_sensor_type": "gas",
        "sensor_type": "vapor_pressure_deficit",
        "value_type": "float",
        "unit": "kPa",
        "norm_range": [0.0, 5.0],
        "physical_quantity": "vapor_pressure_deficit",
        "syntax_directives": [
            "Vapor pressure deficit (VPD) sensor measures the difference between saturation and actual vapor pressure.",
            "It is a key indicator of plant water stress and atmospheric dryness.",
            "Measurements are reported in kilopascals (kPa).",
            "VPD data aids in tailoring irrigation schedules and mitigating drought effects.",
            "Accurate VPD calculations are crucial for optimizing crop water management."
        ]
    },
    {
        "uid": "sound",
        "base_sensor_type": "acoustic",
        "sensor_type": "sound",
        "value_type": "float",
        "unit": "dB",
        "norm_range": [30.0, 120.0],
        "physical_quantity": "ambient_sound_level",
        "syntax_directives": [
            "Sound sensor captures acoustic levels in the surrounding environment.",
            "It is essential for monitoring noise pollution in agricultural and urban areas.",
            "Measurements are provided in decibels (dB).",
            "The sensor aids in assessing the impact of environmental noise on both crops and livestock.",
            "Continuous acoustic monitoring supports improved environmental management."
        ]
    },
    {
        "uid": "flow",
        "base_sensor_type": "fluid",
        "sensor_type": "flow",
        "value_type": "float",
        "unit": "L/min",
        "norm_range": [0.0, 500.0],
        "physical_quantity": "fluid_flow_rate",
        "syntax_directives": [
            "Flow sensor measures the rate at which a fluid passes through a system.",
            "It is critical for managing irrigation systems and water distribution networks.",
            "Output is expressed in liters per minute (L/min).",
            "Real-time flow data supports efficient resource allocation and leak detection.",
            "Accurate monitoring improves pump management and operational safety."
        ]
    },
    {
        "uid": "level",
        "base_sensor_type": "mechanical",
        "sensor_type": "level",
        "value_type": "float",
        "unit": "m",
        "norm_range": [0.0, 5.0],
        "physical_quantity": "liquid_level",
        "syntax_directives": [
            "Level sensor measures the height of a liquid or material within a container or reservoir.",
            "It is essential for monitoring storage tanks and fluid inventory in systems.",
            "Measurements are reported in meters (m).",
            "Data assists in preventing overflows and optimizing refilling operations.",
            "Reliable level measurements enhance process safety and resource management."
        ]
    },
    {
        "uid": "position",
        "base_sensor_type": "positional",
        "sensor_type": "position",
        "value_type": "float", # Note: Original code mentioned encoding as triplet, but value_type remains float here
        "unit": "degrees",
        "norm_range": [0.0, 360.0],
        "physical_quantity": "angular_position",
        "syntax_directives": [
            "Position sensor measures the angular orientation of an object.",
            "It encodes its output as a triplet: raw value, cosine, and sine of the angle (val_cos_sin).",
            "This directional encoding ensures smooth interpolation of angular data.",
            "Measurements are reported in degrees over a 0° to 360° range.",
            "Accurate positional data is vital for precise alignment and motion tracking."
        ]
    },
    {
        "uid": "velocity",
        "base_sensor_type": "positional",
        "sensor_type": "velocity",
        "value_type": "float", # Note: Original code mentioned encoding as triplet, but value_type remains float here
        "unit": "m/s",
        "norm_range": [0.0, 100.0],
        "physical_quantity": "linear_velocity",
        "syntax_directives": [
            "Velocity sensor measures the speed and direction of movement.",
            "It outputs its directional data as a triplet: raw value, cosine, and sine of the angle (val_cos_sin).",
            "This encoding enables accurate representation of vector quantities.",
            "Measurements are provided in meters per second (m/s).",
            "Reliable velocity data supports dynamic process control and motion analysis."
        ]
    },
    {
        "uid": "acceleration",
        "base_sensor_type": "positional",
        "sensor_type": "acceleration",
        "value_type": "float", # Note: Original code mentioned encoding as triplet, but value_type remains float here
        "unit": "m/s²",
        "norm_range": [0.0, 20.0],
        "physical_quantity": "linear_acceleration",
        "syntax_directives": [
            "Acceleration sensor measures the rate of change in velocity over time.",
            "It encodes directional information as a triplet: raw value, cosine, and sine of the angle (val_cos_sin).",
            "Such encoding is essential for capturing vector components of acceleration.",
            "Measurements are expressed in meters per second squared (m/s²).",
            "Accurate acceleration data is crucial for monitoring dynamic system behaviors and shocks."
        ]
    },
    {
        "uid": "direct_radiation",
        "base_sensor_type": "radiation",
        "sensor_type": "direct_radiation",
        "value_type": "float",
        "unit": "W/m²",
        "norm_range": [0.0, 1400.0],
        "physical_quantity": "direct_solar_radiation",
        "syntax_directives": [
            "Direct radiation sensor exclusively measures solar energy received directly from the sun.",
            "It excludes diffused and scattered components of sunlight.",
            "Values are reported in watts per square meter (W/m²).",
            "The sensor is pivotal for accurate solar irradiance and energy balance studies.",
            "Data supports precise agricultural and photovoltaic applications."
        ]
    },
    {
        "uid": "min_temperature",
        "base_sensor_type": "temperature",
        "sensor_type": "min_temperature",
        "value_type": "float",
        "unit": "celsius",
        "norm_range": [-10.0, 40.0],
        "physical_quantity": "daily_min_temperature",
        "syntax_directives": [
            "Min temperature sensor logs the lowest air temperature during a designated period.",
            "It helps identify potential cold stress and frost conditions.",
            "Measurements are provided in degrees Celsius.",
            "Accurate readings support frost forecasting and risk mitigation in agriculture.",
            "Continuous monitoring aids in developing adaptive crop protection strategies."
        ]
    },
    {
        "uid": "wet_temperature",
        "base_sensor_type": "temperature",
        "sensor_type": "wet_temperature",
        "value_type": "float",
        "unit": "celsius",
        "norm_range": [-10.0, 35.0],
        "physical_quantity": "wet_temperature",
        "syntax_directives": [
            "Wet temperature sensor measures the temperature of a wetted thermometer bulb.",
            "It is key for deriving the wet-bulb temperature critical in humidity and evaporation studies.",
            "Readings are expressed in degrees Celsius.",
            "The sensor supports calculations for evaporation rates and cooling processes.",
            "Accurate wet temperature data assists in effective climate control and irrigation planning."
        ]
    },
    {
        "uid": "wind_direction",
        "base_sensor_type": "wind",
        "sensor_type": "wind_direction",
        "value_type": "float", # Note: Original code mentioned encoding as triplet, but value_type remains float here
        "unit": "degrees",
        "norm_range": [0.0, 360.0],
        "physical_quantity": "wind_direction",
        "syntax_directives": [
            "Wind direction sensor determines the origin of the wind.",
            "It encodes its output as a triplet: raw value, cosine, and sine of the angle (val_cos_sin).",
            "This method ensures smooth directional interpolation.",
            "Measurements span the full 0° to 360° range.",
            "Data supports weather forecasting and precise agricultural planning."
        ]
    },
    {
        "uid": "gust_wind_direction",
        "base_sensor_type": "wind",
        "sensor_type": "gust_wind_direction",
        "value_type": "float", # Note: Original code mentioned encoding as triplet, but value_type remains float here
        "unit": "degrees",
        "norm_range": [0.0, 360.0],
        "physical_quantity": "gust_wind_direction",
        "syntax_directives": [
            "Gust wind direction sensor measures the direction of transient wind gusts.",
            "Its output is encoded as a triplet: raw value, cosine, and sine of the angle (val_cos_sin).",
            "This detailed encoding helps capture rapid directional changes.",
            "The sensor covers the entire circular range from 0° to 360°.",
            "Accurate gust direction data aids in dynamic weather and safety assessments."
        ]
    },
    {
        "uid": "max_1min_wind_speed",
        "base_sensor_type": "wind",
        "sensor_type": "max_1min_wind_speed",
        "value_type": "float",
        "unit": "m/s",
        "norm_range": [0.0, 60.0],
        "physical_quantity": "max_1min_wind_speed",
        "syntax_directives": [
            "Max 1-minute wind speed sensor records the highest wind speed within a one-minute interval.",
            "It is critical for capturing short-term gust extremes.",
            "Measurements are provided in meters per second (m/s).",
            "Data assists in immediate weather and safety evaluations.",
            "Accurate peak wind readings support dynamic load assessments on structures."
        ]
    },
    {
        "uid": "max_10min_wind_speed",
        "base_sensor_type": "wind",
        "sensor_type": "max_10min_wind_speed",
        "value_type": "float",
        "unit": "m/s",
        "norm_range": [0.0, 60.0],
        "physical_quantity": "max_10min_wind_speed",
        "syntax_directives": [
            "Max 10-minute wind speed sensor logs the peak wind speed over a ten-minute period.",
            "It provides insights into sustained wind conditions.",
            "Measurements are reported in meters per second (m/s).",
            "This sensor supports regional weather forecasting and structural analysis.",
            "Extended period monitoring facilitates risk management under prolonged wind events."
        ]
    },
    {
        "uid": "time_ending_max_10min_wind_speed",
        "base_sensor_type": "wind",
        "sensor_type": "time_ending_max_10min_wind_speed",
        "value_type": "float",
        "unit": "seconds",
        "norm_range": [0.0, 86400.0],
        "physical_quantity": "time_ending_max_10min_wind_speed",
        "syntax_directives": [
            "This sensor records the time at which the 10-minute maximum wind speed occurred.",
            "It provides temporal context critical for wind event analysis.",
            "Measurements are reported in seconds relative to a reference time.",
            "Data is vital for correlating wind speed peaks with weather events.",
            "Accurate timing information supports detailed meteorological assessments."
        ]
    },
    {
        "uid": "gust_wind_speed",
        "base_sensor_type": "wind",
        "sensor_type": "gust_wind_speed",
        "value_type": "float",
        "unit": "m/s",
        "norm_range": [0.0, 60.0],
        "physical_quantity": "gust_wind_speed",
        "syntax_directives": [
            "Gust wind speed sensor measures the instantaneous speed of wind gusts.",
            "It is essential for capturing brief spikes in wind velocity.",
            "Measurements are provided in meters per second (m/s).",
            "The sensor supports real-time hazard assessments and dynamic modeling.",
            "Accurate gust speed data is critical for structural safety and weather forecasting."
        ]
    },
    {
        "uid": "std_wind_direction",
        "base_sensor_type": "wind",
        "sensor_type": "std_wind_direction",
        "value_type": "float", # Note: Original code mentioned encoding as triplet, but value_type remains float here
        "unit": "degrees",
        "norm_range": [0.0, 180.0],
        "physical_quantity": "std_wind_direction",
        "syntax_directives": [
            "This sensor computes the standard deviation of wind direction measurements.",
            "It encodes the variability as a triplet: raw value, cosine, and sine of the computed angle (val_cos_sin).",
            "This directional encoding helps quantify turbulence and spread in wind patterns.",
            "Measurements aid in evaluating the consistency of wind direction over time.",
            "Accurate directional variability is critical for risk assessments in wind-sensitive operations."
        ]
    },
    {
        "uid": "rainfall",
        "base_sensor_type": "hydrological",
        "sensor_type": "rainfall",
        "value_type": "float",
        "unit": "mm",
        "norm_range": [0.0, 500.0],
        "physical_quantity": "rainfall",
        "syntax_directives": [
            "Rainfall sensor measures the accumulated precipitation over a set period.",
            "It is essential for hydrological assessments and irrigation planning.",
            "Measurements are reported in millimeters (mm).",
            "Data supports flood risk evaluation and water resource management.",
            "Accurate rainfall monitoring is key to sustainable agricultural practices."
        ]
    },
    {
        "uid": "soil_ph",
        "base_sensor_type": "chemical",
        "sensor_type": "soil_ph",
        "value_type": "float",
        "unit": "pH",
        "norm_range": [3.5, 9.0],
        "physical_quantity": "soil_ph",
        "syntax_directives": [
            "Soil pH sensor specifically measures the acidity or alkalinity of soil.",
            "It is critical for assessing soil fertility and nutrient availability.",
            "Values are unitless and reported on a logarithmic scale.",
            "Accurate soil pH data supports precision fertilization and amendment planning.",
            "Monitoring soil pH helps optimize crop production and sustainable land management."
        ]
    }
]