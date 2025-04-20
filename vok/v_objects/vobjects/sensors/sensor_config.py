


class SensorConfig:

    # ------------------------------------------------------------------
    # Aliases from station CSV header (all resolve to existing sensors)
    sensor_type_info_map = {
        'air_temp_high_24_hour_set_1': 'air_temperature',
        'air_temp_high_6_hour_set_1': 'air_temperature',
        'air_temp_low_24_hour_set_1': 'air_temperature',
        'air_temp_low_6_hour_set_1': 'air_temperature',
        'air_temp_set_1': 'air_temperature',
        'air_temperature': 'air_temperature',
        'altimeter_set_1': 'pressure',
        'ceiling_set_1': 'cloud_ceiling',
        'cloud_layer_1_code_set_1': 'cloud_layer_code',
        'cloud_layer_1_set_1d': 'cloud_layer',
        'cloud_layer_2_code_set_1': 'cloud_layer_code',
        'cloud_layer_2_set_1d': 'cloud_layer',
        'cloud_layer_3_code_set_1': 'cloud_layer_code',
        'cloud_layer_3_set_1d': 'cloud_layer',
        'cloud_ceiling': 'cloud_ceiling',
        'cloud_layer': 'cloud_layer',
        'cloud_layer_code': 'cloud_layer_code',
        'dew_point': 'dew_point',
        'dew_point_temperature_set_1': 'dew_point',
        'dew_point_temperature_set_1d': 'dew_point',
        'direction': 'wind_direction',
        'ground_temperature': 'ground_temperature',
        'gust_wind_direction': 'wind_direction',
        'gust_wind_speed_m_s': 'wind_speed',
        'heat_index_set_1d': 'air_temperature',
        'humidity': 'relative_humidity',
        'maximum_10_minutes_wind_speed_m_s': 'wind_speed',
        'maximum_1_minute_wind_speed_m_s': 'wind_speed',
        'maximum_temperature_c': 'air_temperature',
        'metar': 'metar',
        'metar_set_1': 'metar',
        'minimum_temperature_c': 'air_temperature',
        'peak_wind_direction_set_1': 'wind_direction',
        'peak_wind_speed_set_1': 'wind_speed',
        'precip_accum_24_hour_set_1': 'precipitation',
        'precip_accum_one_hour_set_1': 'precipitation',
        'precip_accum_six_hour_set_1': 'precipitation',
        'precip_accum_three_hour_set_1': 'precipitation',
        'precipitation': 'precipitation',
        'pressure': 'pressure',
        'pressure_at_station_level_hpa': 'pressure',
        'pressure_change_code': 'pressure_change_code',
        'pressure_change_code_set_1': 'pressure_change_code',
        'pressure_set_1d': 'pressure',
        'pressure_tendency_set_1': 'pressure',
        'rainfall_mm': 'precipitation',
        'relative_humidity': 'relative_humidity',
        'relative_humidity_set_1': 'relative_humidity',
        'rh': 'relative_humidity',
        'sea_level_pressure_set_1': 'pressure',
        'sea_level_pressure_set_1d': 'pressure',
        'standard_deviation_wind_direction': 'wind_direction',
        'temp_c': 'air_temperature',
        'temperature_c': 'air_temperature',
        'vpd': 'vpd',
        'visibility': 'visibility',
        'visibility_set_1': 'visibility',
        'weather_code': 'weather_code',
        'weather_cond_code_set_1': 'weather_code',
        'weather_condition_set_1d': 'weather_code',
        'weather_summary': 'weather_summary',
        'weather_summary_set_1d': 'weather_summary',
        'wet_temperature_c': 'air_temperature',
        'wind': 'wind_speed',
        'wind_cardinal_direction_set_1d': 'wind_direction',
        'wind_direction': 'wind_direction',
        'wind_direction_set_1': 'wind_direction',
        'wind_gust_set_1': 'wind_speed',
        'wind_mps': 'wind_speed',
        'wind_speed': 'wind_speed',
        'wind_speed_m_s': 'wind_speed',
        'wind_speed_set_1': 'wind_speed',
        'wind_chill_set_1d': 'air_temperature',
    }

    # ------------------------------------------------------------------



 

    # detailed configuration for each specific sensor type
    sensor_type_info = [
        {
            'uid': 'sensor_type_air_temperature',
            'sensor_name': 'air_temperature',
            'base_sensor_type': 'temperature',
            'unit': 'celsius',
            'data_format': 'float',
            'single_data_size': 1,
            'syntax_directives': [
                'Measures ambient atmospheric temperature at a specified height.',
                'Captures variations due to solar radiation and shading.',
                'Reflects impacts of wind and local microclimate factors.'
            ],
            'measurement_range': {'min': -40.0, 'max': 85.0},
            'norm_range': [-5.0, 50.0],
        },
        {
            'uid': 'sensor_type_relative_humidity',
            'sensor_name': 'relative_humidity',
            'base_sensor_type': 'humidity',
            'unit': 'percent',
            'data_format': 'float',
            'single_data_size': 1,
            'syntax_directives': [
                'Measures the amount of water vapor in the air as a percentage.',
                'Reflects atmospheric moisture content relative to temperature and pressure.',
                'Important for assessing comfort and forecasting precipitation.'
            ],
            'measurement_range': {'min': 0.0, 'max': 100.0},
            'norm_range': [0.0, 100.0],
        },
        {
            'uid': 'sensor_type_ground_temperature',
            'sensor_name': 'ground_temperature',
            'base_sensor_type': 'temperature',
            'unit': 'celsius',
            'data_format': 'float',
            'single_data_size': 1,
            'syntax_directives': [
                'Measures temperature at the soil surface.',
                'Captures heat transfer between ground and air.',
                'Influences soil biological and chemical processes.'
            ],
            'measurement_range': {'min': -20.0, 'max': 60.0},
            'norm_range': [-5.0, 50.0],
        },
        {
            'uid': 'sensor_type_dew_point',
            'sensor_name': 'dew_point',
            'base_sensor_type': 'temperature',
            'unit': 'celsius',
            'data_format': 'float',
            'single_data_size': 1,
            'syntax_directives': [
                'The temperature at which air becomes fully saturated with moisture.',
                'Indicates the onset of condensation and fog.',
                'Closely related to humidity and atmospheric pressure.'
            ],
            'measurement_range': {'min': -40.0, 'max': 30.0},
            'norm_range': [-5.0, 30.0],
        },
        {
            'uid': 'sensor_type_vpd',
            'sensor_name': 'vpd',
            'base_sensor_type': 'pressure',
            'unit': 'kilopascal',
            'data_format': 'float',
            'single_data_size': 1,
            'syntax_directives': [
                'Vapor Pressure Deficit measures the pressure difference between saturated and actual vapor pressure.',
                'An important indicator of plant water stress and transpiration rates.',
                'Combines temperature and humidity information to quantify atmospheric drying power.'
            ],
            'measurement_range': {'min': 0.2, 'max': 3.0},  # typical range 0.2–3 kPa
            'norm_range': [0.2, 3.0],  # normalization range for model inputs
        },
        {
            "uid": "sensor_type_wind_speed",
            "sensor_name": "wind_speed",
            "base_sensor_type": "wind",
            "unit": "meter_per_second",
            "data_format": "float",
            "single_data_size": 1,
            "syntax_directives": [...],
            "measurement_range": {"min": 0.0, "max": 60.0},
            "norm_range": [0.0, 30.0],
        },
        {
            "uid": "sensor_type_wind_direction",
            "sensor_name": "wind_direction",
            "base_sensor_type": "wind",
            "unit": "degrees",           # raw unit pre‑encoding
            "data_format": "cyclic_float_pair",
            "single_data_size": 2,       # sin, cos
            "encoding": ["sin", "cos"],
            "syntax_directives": [...],
            "measurement_range": {"min": 0.0, "max": 360.0},
            "norm_range": [0.0, 360.0],  # before trig transform
        },
        # ──────────────────────────────────────────────────────────────
    # Add‑on sensor_type_info items (10 total)
    # ──────────────────────────────────────────────────────────────

        # ───────── Atmospheric Pressure ─────────
        {
            'uid': 'sensor_type_pressure',
            'sensor_name': 'pressure',
            'base_sensor_type': 'pressure',
            'unit': 'hectopascal',          # add to unit_info if not present
            'data_format': 'float',
            'single_data_size': 1,
            'syntax_directives': [
                'Measures barometric pressure at the station sensor height.',
                'Essential for synoptic‑scale weather analysis and altitude correction.',
                'Used in evapotranspiration and psychrometric calculations.'
            ],
            'measurement_range': {'min': 870.0, 'max': 1080.0},   # hPa extremes
            'norm_range': [980.0, 1040.0],
        },

        # ───────── Precipitation (Accumulated) ─────────
        {
            'uid': 'sensor_type_precipitation',
            'sensor_name': 'precipitation',
            'base_sensor_type': 'precipitation',
            'unit': 'millimeter',           # add to unit_info if not present
            'data_format': 'float',
            'single_data_size': 1,
            'syntax_directives': [
                'Cumulative liquid‑equivalent precipitation since last reset.',
                'Drives soil‑water balance and runoff models.',
                'Helps calibrate irrigation scheduling and flood‑risk assessment.'
            ],
            'measurement_range': {'min': 0.0, 'max': 500.0},
            'norm_range': [0.0, 200.0],
        },

        # ───────── Visibility ─────────
        {
            'uid': 'sensor_type_visibility',
            'sensor_name': 'visibility',
            'base_sensor_type': 'visibility',
            'unit': 'meter',
            'data_format': 'float',
            'single_data_size': 1,
            'syntax_directives': [
                'Prevailing horizontal visibility along the sensor’s optical path.',
                'Affected by fog, haze, smoke, rain, and blowing dust.',
                'Critical for transportation safety and air‑quality alerts.'
            ],
            'measurement_range': {'min': 0.0, 'max': 20000.0},     # up to 20 km
            'norm_range': [0.0, 10000.0],
        },

        # ───────── Cloud Ceiling ─────────
        {
            'uid': 'sensor_type_cloud_ceiling',
            'sensor_name': 'cloud_ceiling',
            'base_sensor_type': 'cloud',
            'unit': 'meter',
            'data_format': 'float',
            'single_data_size': 1,
            'syntax_directives': [
                'Height above ground of the lowest cloud layer covering ≥ 5/8 of the sky.',
                'Derived from laser ceilometers or human observations.',
                'Important for aviation minima and solar‑irradiance forecasting.'
            ],
            'measurement_range': {'min': 0.0, 'max': 20000.0},
            'norm_range': [0.0, 5000.0],
        },

        # ───────── Cloud Layer Fractional Cover ─────────
        {
            'uid': 'sensor_type_cloud_layer',
            'sensor_name': 'cloud_layer',
            'base_sensor_type': 'cloud',
            'unit': 'okta',                 # 0–8 eighths of sky; add to unit_info
            'data_format': 'integer',
            'single_data_size': 1,
            'syntax_directives': [
                'Total cloud cover for a specific layer expressed in oktas (eighths).',
                'Provides finer detail than overall sky condition for multilayer decks.',
                'Supports radiative‑transfer and energy‑balance models.'
            ],
            'measurement_range': {'min': 0, 'max': 8},
            'norm_range': [0, 8],
        },

        # ───────── Cloud Layer Type Code ─────────
        {
            'uid': 'sensor_type_cloud_layer_code',
            'sensor_name': 'cloud_layer_code',
            'base_sensor_type': 'cloud',
            'unit': 'none',
            'data_format': 'integer',
            'single_data_size': 1,
            'syntax_directives': [
                'WMO code (0–9) identifying the dominant cloud‑layer form (e.g., cumulus, nimbostratus).',
                'Enables automated interpretation of METAR/SYNOP strings.',
                'Useful for cloud‑type climatology and photovoltaic performance studies.'
            ],
            'measurement_range': {'min': 0, 'max': 9},
            'norm_range': [0, 9],
        },

        # ───────── Pressure‑Change Tendency Code ─────────
        {
            'uid': 'sensor_type_pressure_change_code',
            'sensor_name': 'pressure_change_code',
            'base_sensor_type': 'pressure',
            'unit': 'none',
            'data_format': 'integer',
            'single_data_size': 1,
            'syntax_directives': [
                'Three‑hour barometric‑tendency code (0–8) per WMO table 0200.',
                'Classifies rising, falling, or steady pressure trends.',
                'Aids rapid cyclone‑development and storm‑warning systems.'
            ],
            'measurement_range': {'min': 0, 'max': 8},
            'norm_range': [0, 8],
        },

        # ───────── Weather Code ─────────
        {
            'uid': 'sensor_type_weather_code',
            'sensor_name': 'weather_code',
            'base_sensor_type': 'weather',
            'unit': 'none',
            'data_format': 'integer',
            'single_data_size': 1,
            'syntax_directives': [
                'Present‑weather WMO code (00–99) representing phenomena like rain, snow, dust, or thunderstorms.',
                'Facilitates machine‑readable categorisation of METAR/SYNOP observations.',
                'Supplies categorical input for hazard and comfort‑index models.'
            ],
            'measurement_range': {'min': 0, 'max': 99},
            'norm_range': [0, 99],
        },

        # ───────── Weather Summary (Human‑Readable) ─────────
        {
            'uid': 'sensor_type_weather_summary',
            'sensor_name': 'weather_summary',
            'base_sensor_type': 'text_report',
            'unit': 'none',
            'data_format': 'string',
            'single_data_size': 1,
            'syntax_directives': [
                'Concise natural‑language syntax_directives of current weather (e.g., “Light rain, mist”).',
                'Derived from decoded METAR reports or local observers.',
                'Useful for dashboards and end‑user alerts where numeric codes are unsuitable.'
            ],
            'measurement_range': None,
            'norm_range': None,
        },

        # ───────── METAR Raw Report ─────────
        {
            'uid': 'sensor_type_metar',
            'sensor_name': 'metar',
            'base_sensor_type': 'text_report',
            'unit': 'none',
            'data_format': 'string',
            'single_data_size': 1,
            'syntax_directives': [
                'Full raw METAR string including station identifier, time, wind, visibility, weather, clouds, temperature, pressure, and remarks.',
                'Serves as an immutable text snapshot for forensic replay and regulatory compliance.',
                'Parsed downstream to populate structured sensor fields in this configuration.'
            ],
            'measurement_range': None,
            'norm_range': None,
        },


    ]

    base_sensor_type_info = [
        {
            'uid': 'base_sensor_type_temperature',
            'sensor_type': 'temperature',
            'physical_quantity': 'Temperature',
            'syntax_directives': [
                'Temperature measures thermal energy in the environment.',
                'Canonical units: Celsius (°C), Fahrenheit (°F), Kelvin (K).',
                'Applicable to air, ground, dew‑point, etc.'
            ],
            'allowed_units': ['celsius', 'fahrenheit', 'kelvin']
        },
        {
            'uid': 'base_sensor_type_humidity',
            'sensor_type': 'humidity',
            'physical_quantity': 'Relative Humidity',
            'syntax_directives': [
                'Humidity expresses water‑vapor content in the air.',
                'Canonical unit: percent (%) of saturation.',
                'Drives dew‑point and VPD calculations.'
            ],
            'allowed_units': ['percent']
        },
        {
            'uid': 'base_sensor_type_pressure',
            'sensor_type': 'pressure',
            'physical_quantity': 'Pressure',
            'syntax_directives': [
                'Atmospheric or vapor pressure measurements.',
                'Canonical units: pascal (Pa) and kilopascal (kPa).',
                'Includes sea‑level pressure, altimeter, VPD.'
            ],
            'allowed_units': ['pascal', 'kilopascal']
        },
        {
            'uid': 'base_sensor_type_wind_speed',
            'sensor_type': 'wind_speed',
            'physical_quantity': 'Wind Speed',
            'syntax_directives': [
                'Linear velocity of air movement.',
                'Canonical unit: meters per second (m_s).',
                'Covers instantaneous, gust, and peak speeds.'
            ],
            'allowed_units': ['m_s']
        },
        {
            'uid': 'base_sensor_type_wind_direction',
            'sensor_type': 'wind_direction',
            'physical_quantity': 'Wind Direction',
            'syntax_directives': [
                'Compass direction from which wind originates.',
                'Measured in degrees but encoded as cyclic sin/cos.',
                '0° = north, increasing clockwise to 360°.'
            ],
            'allowed_units': ['degrees']
        },
        {
            'uid': 'base_sensor_type_wind',
            'sensor_type': 'wind',
            'physical_quantity': 'Wind',
            'syntax_directives': [
                'Encompasses both linear wind speed and compass direction.',
                'Crucial for mass‑transfer, evapotranspiration, and dispersion models.',
                'Feeds derivative metrics such as wind‑run and turbulence intensity.'
            ],
            'allowed_units': ['meter_per_second', 'degrees']
        },

        # ───────── Precipitation ─────────
        {
            'uid': 'base_sensor_type_precipitation',
            'sensor_type': 'precipitation',
            'physical_quantity': 'Precipitation',
            'syntax_directives': [
                'Measures liquid‑equivalent water delivered to the surface.',
                'Includes both instantaneous rate and cumulative depth.',
                'Key driver for hydrology, irrigation, and flood‑risk analytics.'
            ],
            'allowed_units': ['millimeter', 'millimeter_per_hour']
        },

        # ───────── Visibility ─────────
        {
            'uid': 'base_sensor_type_visibility',
            'sensor_type': 'visibility',
            'physical_quantity': 'Visibility',
            'syntax_directives': [
                'Horizontal optical range under prevailing conditions.',
                'Impacted by fog, haze, dust, precipitation, and smoke.',
                'Supports transport‑safety indices and air‑quality assessments.'
            ],
            'allowed_units': ['meter']
        },

        # ───────── Cloud ─────────
        {
            'uid': 'base_sensor_type_cloud',
            'sensor_type': 'cloud',
            'physical_quantity': 'Cloud Properties',
            'syntax_directives': [
                'Describes cloud cover, height, and morphological class.',
                'Combines fractional sky‑cover (oktas) with base height metrics.',
                'Important for solar‑irradiance, aviation, and climate models.'
            ],
            'allowed_units': ['okta', 'meter']
        },

        # ───────── Weather (coded) ─────────
        {
            'uid': 'base_sensor_type_weather',
            'sensor_type': 'weather',
            'physical_quantity': 'Present Weather Code',
            'syntax_directives': [
                'Categorical representation of observed weather phenomena.',
                'Uses WMO 00–99 code set (e.g., rain, snow, dust, thunderstorm).',
                'Simplifies ingestion into alerting and hazard‑classification systems.'
            ],
            'allowed_units': ['none']
        },

        # ───────── Text Report ─────────
        {
            'uid': 'base_sensor_type_text_report',
            'sensor_type': 'text_report',
            'physical_quantity': 'Encoded Weather Report',
            'syntax_directives': [
                'Free‑text or encoded weather strings such as METAR or SPECI.',
                'Preserves full observational context for forensic replay.',
                'Parsed downstream into structured fields when needed.'
            ],
            'allowed_units': ['none']
        },
    ]



    # import pprint
    # import re

    # # --- existing alias → canonical‑sensor map -------------------------

    # # ───────────────────────────────────────────────────────────────────
    # # helper: pull fields out of the header string
    # # ───────────────────────────────────────────────────────────────────
    # def _decode_header(header: str, canonical: str) -> dict:
    #     """
    #     Return a metadata dict for a single CSV header.
    #     Unknown values are normalised to None.
    #     """
    #     info = {
    #         "sensor_type"     : canonical,      # resolved name
    #         "statistic"       : None,           # max / min / gust / peak / accum …
    #         "duration_hours"  : None,           # integer
    #         "duration_minutes": None,           # integer
    #         "accumulation"    : False,          # True if 'accum' found
    #         "layer"           : None,           # cloud_layer_X  -> X
    #         "set_index"       : None,           # …_set_N[_d]
    #         "unit_suffix"     : None,           # _c, _m_s, _mm, _hpa …
    #         "original_key"    : header,         # keep the raw text for reference
    #     }

    #     # 1) layer number (cloud layers only)
    #     m = re.search(r"cloud_layer_(\d)", header)
    #     if m:
    #         info["layer"] = int(m.group(1))

    #     # 2) statistics keywords
    #     if   "accum"   in header: info["statistic"], info["accumulation"] = "accum", True
    #     elif "high"    in header or "maximum" in header or "peak" in header: info["statistic"] = "max"
    #     elif "low"     in header or "minimum" in header:                       info["statistic"] = "min"
    #     elif "gust"    in header:                                              info["statistic"] = "gust"

    #     # 3) duration (hours / minutes)
    #     if m := re.search(r"(\d+)_hour", header):
    #         info["duration_hours"] = int(m.group(1))
    #     if m := re.search(r"(\d+)_minutes?", header):
    #         info["duration_minutes"] = int(m.group(1))
    #     # human‑worded shortcuts
    #     if "one_hour"   in header: info["duration_hours"] = 1
    #     if "three_hour" in header: info["duration_hours"] = 3
    #     if "six_hour"   in header: info["duration_hours"] = 6

    #     # 4) set index
    #     if m := re.search(r"set_(\d+)d?", header):
    #         info["set_index"] = int(m.group(1))

    #     # 5) unit suffix hint (very coarse)
    #     if m := re.search(r"_(c|m_s|mm|hpa)$", header):
    #         info["unit_suffix"] = m.group(1)

    #     return info



    # if __name__ == '__main__':
    # # ───────────────────────────────────────────────────────────────────
    # # build the new dictionary
    # # ───────────────────────────────────────────────────────────────────
    #     sensor_id_info = {
    #         alias: _decode_header(alias, canonical)
    #         for alias, canonical in sensor_type_info_map.items()
    #     }


    #     nsi = {}
    #     # quick sanity check
    #     for k in sensor_id_info:
    #         print(k, "→", sensor_id_info[k])
    #         nsi[k] = sensor_id_info[k]

    #     pprint.pp(nsi)