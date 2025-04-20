
import pprint

import numpy as np
from dawn_vok.db.mongo_utils import MongoUtils
from dawn_vok.utils.id_utils import IDUtils
from dawn_vok.vok.dobjects.d_object import DObject
from dawn_vok.vok.dobjects.dobj.config.sensor_type_conf import SensorTypeConfig
from dawn_vok.vok.v_objects.vok_object import VOKObject
# from dawn_vok.vok.embedding.syntax.syntax_db_builder import SyntaxDBBuilder


class SensorType(VOKObject):
    @classmethod
    def get_db_name(cls):
        return 'meta_data'
    
    @classmethod
    def get_collection_name(cls):
        return 'sensor_type'
    
    def __init__(self, uid, di={}):
        self.base_sensor_type = di.get('base_sensor_type', None)
        self.sensor_type = di.get('sensor_type', None)
        self.unit = di.get('unit', None)
        self.physical_quantity = di.get('physical_quantity', None)
        self.value_type = di.get('value_type', None)
        provider_type = di.get('provider_type', None)   
        system_uid = IDUtils.get_sensor_type_id(provider_type)
        super().__init__(obj_type='sensor_type', uid=uid, system_uid=system_uid)
        self.populate_from_dict(di)

    
      
        
    def get_vocab(self):
        ret = super().get_vocab()   
        if self.value_type:
            ret.append("the value type of the sensor is " + self.value_type)
        if self.base_sensor_type:
            ret.append("the base sensor type of the sensor is " + self.base_sensor_type)
        if self.sensor_type:
            ret.append("the sensor type of the sensor is " + self.sensor_type)
        if self.unit:
            ret.append("the unit of the sensor is " + self.unit)
        if self.physical_quantity:
            ret.append("the physical quantity of the sensor is " + self.physical_quantity)
        return ret

    def to_dict(self):
        ret = super().to_dict()
        ret['base_sensor_type'] = self.base_sensor_type
        ret['sensor_type'] = self.sensor_type
        ret['unit'] = self.unit
        ret['physical_quantity'] = self.physical_quantity
        ret['value_type'] = self.value_type
        return ret
    
    def populate_from_dict(self, d):    
        super().populate_from_dict(d)
        self.value_type = d.get('value_type', self.value_type)
        self.base_sensor_type = d.get('base_sensor_type', self.base_sensor_type)
        self.sensor_type = d.get('sensor_type', self.sensor_type)
        self.unit = d.get('unit', self.unit)
        self.physical_quantity = d.get('physical_quantity', self.physical_quantity)
        return self
    
    
    @classmethod
    def create_all_sensor_types(cls):
        sensor_types = []
        for st in SensorTypeConfig.type_info:
            uid = st['uid']
            sensor_types.append(SensorType(uid, st))


        MongoUtils.update_many(cls.get_db_name(), cls.get_collection_name(), sensor_types)
        return sensor_types
    
    @classmethod
    def gather_full_vocab(cls):
        sensor_types = cls.get_all()
        print(sensor_types)
        vocab = []
        for st in sensor_types:
            vocab.extend(st.get_vocab())
        vocab = list(set(vocab))
        return vocab
    
# if __name__ == '__main__':
#     SensorType.create_all_sensor_types()
#     vocab = SensorType.gather_full_vocab()
#     builder = SyntaxDBBuilder()
#     builder.build_syntax_db(vocab)
#     builder.save_to_db()
    # trainer = SyntaxEmbeddingReductionTrainer()
    # trainer.update_embeddings(emb_size=32, orig_scheme_id='full_embedding', out_scheme_id='reduced_32')
    # trainer.save_model()
    # trainer = SyntaxEmbeddingReductionTrainer(short_emb_size=16)
    # trainer.update_embeddings(emb_size=16, orig_scheme_id='full_embedding', out_scheme_id='reduced_16')
    # trainer.save_model()
    # "air_temperature": {
    #     "sensor_type": "temperature",
    #     "value_type": "float",
    #     "sensor_value_unit": "Celsius",
    #     "sensor_class": "environmental",
    #     "range_expected": [-50.0, 60.0],
    #     "precision": 0.1,
    #     "physical_quantity": "air_heat_content",
    #     "application_context": ["greenhouse", "outdoor", "weather_station"],
    #     "syntax_directives": [
    #         "Air temperature measures the thermal energy of the surrounding environment.",
    #         "This sensor is critical for evaluating plant growth conditions and climate control.",
    #         "It outputs values in degrees Celsius, typically ranging from -50 to 60."
    #     ]
    # },
 #       temperature_sensor = SensorType(
#     uid='air_temperature',
#     base_sensor_type='temperature',
#     sensor_type='temperature',
#     value_type='float',
#     unit='celsius',
#     norm_range=[-5.0, 50.0],
#     physical_quantity='air_heat_content',
#     syntax_directives=[
#         "Air temperature measures the thermal energy of the surrounding environment.",
#         "It outputs values in degrees Celsius, normalized between -50 to 60.",
#         "It continuously monitors diurnal temperature variations, capturing the daily cycle.",
#         "Seasonal shifts in temperature are tracked to analyze yearly climate trends.",
#         "Sensor data supports adaptive strategies in precision agriculture, ensuring optimal plant growth and resource management."
#     ]
# )

#         sensor_types.append(temperature_sensor)
#         humidity_sensor = SensorType(
#     uid='relative_humidity',
#     base_sensor_type='humidity', 
#     sensor_type='humidity',
#     value_type='float',
#     unit='percentage',
#     norm_range=[0.0, 100.0], 
#     physical_quantity='water_vapor_ratio',
#     syntax_directives=[
#         "Relative humidity represents the percentage of water vapor present in the air.",
#         "It outputs values as percentages, normalized between 0 to 100.",
#         "The sensor provides crucial insights into atmospheric moisture for both weather forecasting and climate analysis.",
#         "It monitors daily fluctuations to help understand evaporation and condensation cycles impacting plant transpiration.",
#         "Seasonal variations in humidity levels support irrigation planning and crop water stress management."
#     ]
# )
#         sensor_types.append(humidity_sensor)
#         radiation_sensor = SensorType(
#             uid='radiation',
#             base_sensor_type='radiation',
#             sensor_type='radiation',
#             value_type='float',
#             unit='W/m²',
#             norm_range=[0.0, 1400.0],
#             physical_quantity='solar_energy_intensity',
#             syntax_directives=[
#                 "Radiation measures the amount of solar energy received per square meter.",
#                 "It is crucial for understanding the energy balance in agricultural and environmental systems.",
#                 "This sensor plays a vital role in monitoring plant photosynthetic activity.",
#                 "Values are reported in watts per square meter (W/m²) and influence crop growth analysis.",
#                 "Data helps in assessing both diurnal and seasonal variations in solar intensity."
#             ]
#         )
#         sensor_types.append(radiation_sensor)

#         # Soil Moisture Sensor
#         soil_moisture_sensor = SensorType(
#             uid='soil_moisture',
#             base_sensor_type='moisture',
#             sensor_type='soil_moisture',
#             value_type='float',
#             unit='m³/m³',
#             norm_range=[0.0, 0.6],
#             physical_quantity='volumetric_water_content',
#             syntax_directives=[
#                 "Soil moisture indicates the volumetric water content within the soil.",
#                 "It is essential for effective irrigation and plant water stress management.",
#                 "Accurate measurements support soil water balance and root zone analysis.",
#                 "Data is measured in cubic meters of water per cubic meter of soil (m³/m³).",
#                 "It facilitates real-time decisions in precision agriculture and resource optimization."
#             ]
#         )
#         sensor_types.append(soil_moisture_sensor)

#         # Air Pressure Sensor
#         air_pressure_sensor = SensorType(
#             uid='air_pressure',
#             base_sensor_type='pressure',
#             sensor_type='air_pressure',
#             value_type='float',
#             unit='hPa',
#             norm_range=[800.0, 1100.0],
#             physical_quantity='atmospheric_pressure',
#             syntax_directives=[
#                 "Air pressure represents the atmospheric force exerted per unit area.",
#                 "It is a key parameter for weather prediction and environmental monitoring.",
#                 "The sensor assists in altitude adjustment and meteorological studies.",
#                 "Measurements are recorded in hectopascals (hPa).",
#                 "Understanding pressure variations aids in forecasting storm formation and climate shifts."
#             ]
#         )
#         sensor_types.append(air_pressure_sensor)

#         # Diffused Radiation Sensor
#         diffused_radiation_sensor = SensorType(
#             uid='diffused_radiation',
#             base_sensor_type='radiation',
#             sensor_type='diffused_radiation',
#             value_type='float',
#             unit='W/m²',
#             norm_range=[0.0, 800.0],
#             physical_quantity='scattered_solar_energy',
#             syntax_directives=[
#                 "Diffused radiation quantifies the scattered component of solar energy after atmospheric interaction.",
#                 "It is important for evaluating light conditions in shaded environments.",
#                 "The sensor aids in accurately modeling canopy light distribution and photosynthesis.",
#                 "Measurements are provided in watts per square meter (W/m²).",
#                 "This data is critical for optimizing crop layouts and shade management strategies."
#             ]
#         )
#         sensor_types.append(diffused_radiation_sensor)

#         # Global Radiation Sensor
#         global_radiation_sensor = SensorType(
#             uid='global_radiation',
#             base_sensor_type='radiation',
#             sensor_type='global_radiation',
#             value_type='float',
#             unit='W/m²',
#             norm_range=[0.0, 1400.0],
#             physical_quantity='total_solar_energy',
#             syntax_directives=[
#                 "Global radiation captures the complete solar energy incident on a horizontal surface.",
#                 "It encompasses both direct sunlight and diffused sky radiation.",
#                 "This comprehensive measurement is fundamental for energy balance studies.",
#                 "Data is expressed in watts per square meter (W/m²).",
#                 "It supports assessment of overall solar input for crop production and climatology."
#             ]
#         )
#         sensor_types.append(global_radiation_sensor)

#         # Max Temperature Sensor
#         max_temperature_sensor = SensorType(
#             uid='max_temperature',
#             base_sensor_type='temperature',
#             sensor_type='max_temperature',
#             value_type='float',
#             unit='Celsius',
#             norm_range=[-10.0, 60.0],
#             physical_quantity='daily_max_temperature',
#             syntax_directives=[
#                 "Max temperature logs the highest recorded air temperature within a specified period.",
#                 "It is a crucial metric for detecting heat stress in plants and humans.",
#                 "The sensor output aids in identifying extreme weather conditions.",
#                 "Data is expressed in degrees Celsius for clarity in temperature range.",
#                 "It supports both short-term and long-term climate trend analysis."
#             ]
#         )
#         sensor_types.append(max_temperature_sensor)

#         # Wind Speed Sensor
#         wind_speed_sensor = SensorType(
#             uid='wind_speed',
#             base_sensor_type='wind',
#             sensor_type='wind_speed',
#             value_type='float',
#             unit='m/s',
#             norm_range=[0.0, 60.0],
#             physical_quantity='horizontal_wind_velocity',
#             syntax_directives=[
#                 "Wind speed quantifies the horizontal velocity of moving air.",
#                 "It is essential for understanding weather dynamics and dispersion phenomena.",
#                 "Measurements are critical for calculating evapotranspiration in agricultural models.",
#                 "Sensor readings are provided in meters per second (m/s).",
#                 "This data contributes to wind energy assessments and environmental modeling."
#             ]
#         )
#         sensor_types.append(wind_speed_sensor)
#         sensor_types.append(wind_speed_sensor)
#         co2_sensor = SensorType(
#             uid='co2',
#             base_sensor_type='gas',
#             sensor_type='co2',
#             value_type='float',
#             unit='ppm',
#             norm_range=[300.0, 2000.0],
#             physical_quantity='carbon_dioxide_concentration',
#             syntax_directives=[
#                 "CO2 sensor measures the concentration of carbon dioxide in the air.",
#                 "Relevant for photosynthesis modeling and greenhouse ventilation.",
#                 "Reported in parts per million (ppm)."
#             ]
#         )
#         sensor_types.append(co2_sensor)
#         ph_sensor = SensorType(
#             uid='ph',
#             base_sensor_type='chemical',
#             sensor_type='ph',
#             value_type='float',
#             unit='pH',
#             norm_range=[3.5, 9.0],
#             physical_quantity='hydrogen_ion_activity',
#             syntax_directives=[
#                 "pH represents the acidity or alkalinity of a substance.",
#                 "Essential for soil chemistry, nutrient uptake, and hydroponics.",
#                 "Values are unitless on a logarithmic scale."
#             ]
#         )
#         sensor_types.append(ph_sensor)
#         light_intensity_sensor = SensorType(
#             uid='light_intensity',
#             base_sensor_type='radiation',
#             sensor_type='light_intensity',
#             value_type='float',
#             unit='lux',
#             norm_range=[0, 120000],
#             physical_quantity='visible_light_flux',
#             syntax_directives=[
#                 "Light intensity measures the amount of visible light reaching a surface.",
#                 "Important for plant photoperiodism, shading analysis, and indoor lighting.",
#                 "Reported in lux (lumens per square meter)."
#             ]
#         )
#         sensor_types.append(light_intensity_sensor)
#         # CO2 Sensor
#         co2_sensor = SensorType(
#             uid='co2',
#             base_sensor_type='gas',
#             sensor_type='co2',
#             value_type='float',
#             unit='ppm',
#             norm_range=[300.0, 2000.0],
#             physical_quantity='carbon_dioxide_concentration',
#             syntax_directives=[
#                 "CO2 sensor measures the concentration of carbon dioxide in the air.",
#                 "It is essential for assessing greenhouse gas levels in environmental studies.",
#                 "The sensor data helps optimize ventilation in greenhouses for improved plant health.",
#                 "Measurements are reported in parts per million (ppm), a standard unit for gas concentration.",
#                 "Continuous monitoring supports analysis of seasonal trends and environmental shifts."
#             ]
#         )
#         sensor_types.append(co2_sensor)

#         # pH Sensor
#         ph_sensor = SensorType(
#             uid='ph',
#             base_sensor_type='chemical',
#             sensor_type='ph',
#             value_type='float',
#             unit='pH',
#             norm_range=[3.5, 9.0],
#             physical_quantity='hydrogen_ion_activity',
#             syntax_directives=[
#                 "pH sensor measures the acidity or alkalinity of a substance.",
#                 "It plays a critical role in monitoring soil chemistry and nutrient availability.",
#                 "The device aids in maintaining optimum conditions for hydroponics and plant growth.",
#                 "pH values are reported on a logarithmic scale to indicate hydrogen ion concentration.",
#                 "Regular monitoring supports adjustments in fertilization and ensures soil health."
#             ]
#         )
#         sensor_types.append(ph_sensor)

#         # Light Intensity Sensor
#         light_intensity_sensor = SensorType(
#             uid='light_intensity',
#             base_sensor_type='radiation',
#             sensor_type='light_intensity',
#             value_type='float',
#             unit='lux',
#             norm_range=[0, 120000],
#             physical_quantity='visible_light_flux',
#             syntax_directives=[
#                 "Light intensity sensor measures the amount of visible light reaching a surface.",
#                 "It is crucial for assessing the level of photosynthetically active radiation available to plants.",
#                 "The sensor is used in both indoor lighting management and outdoor sun exposure analysis.",
#                 "Measurements are reported in lux, a standard unit for luminous flux per unit area.",
#                 "Data from this sensor informs shading strategies and optimizes crop yield potential."
#             ]
#         )
#         sensor_types.append(light_intensity_sensor)
#         # Soil Temperature Sensor
#         soil_temperature_sensor = SensorType(
#             uid='soil_temperature',
#             base_sensor_type='temperature',
#             sensor_type='soil_temperature',
#             value_type='float',
#             unit='celsius',
#             norm_range=[-5.0, 45.0],
#             physical_quantity='soil_temperature',
#             syntax_directives=[
#                 "Soil temperature sensor measures the thermal state of the soil profile.",
#                 "It is vital for understanding root zone dynamics and seed germination.",
#                 "The sensor captures both daily and seasonal temperature fluctuations.",
#                 "Accurate soil temperature data helps in optimizing irrigation and planting schedules.",
#                 "Monitoring soil temperature is key to assessing soil health and crop performance."
#             ]
#         )
#         sensor_types.append(soil_temperature_sensor)

#         # Soil Electrical Conductivity (EC) Sensor
#         soil_ec_sensor = SensorType(
#             uid='soil_ec',
#             base_sensor_type='electrical',
#             sensor_type='soil_ec',
#             value_type='float',
#             unit='dS/m',
#             norm_range=[0.0, 5.0],
#             physical_quantity='soil_electrical_conductivity',
#             syntax_directives=[
#                 "Soil EC sensor measures the electrical conductivity of the soil solution.",
#                 "It provides insights into the soil’s salinity and ion concentration.",
#                 "High EC values may indicate excess salt, which can affect plant growth.",
#                 "The sensor aids in managing irrigation and fertilizer application strategies.",
#                 "Monitoring EC helps ensure sustainable soil management and optimal crop yield."
#             ]
#         )
#         sensor_types.append(soil_ec_sensor)

#         # Soil Organic Matter Sensor
#         soil_organic_matter_sensor = SensorType(
#             uid='soil_organic_matter',
#             base_sensor_type='chemical',
#             sensor_type='soil_organic_matter',
#             value_type='float',
#             unit='percentage',
#             norm_range=[0.0, 20.0],
#             physical_quantity='soil_organic_matter',
#             syntax_directives=[
#                 "This sensor quantifies the organic matter content in the soil.",
#                 "Organic matter is essential for improving soil fertility and water retention.",
#                 "It plays a crucial role in nutrient cycling and soil structure.",
#                 "Accurate readings support sustainable land management and organic amendment practices.",
#                 "Data from the sensor helps track soil quality and long-term ecological health."
#             ]
#         )
#         sensor_types.append(soil_organic_matter_sensor)

#         # Soil Nitrogen Sensor
#         soil_nitrogen_sensor = SensorType(
#             uid='soil_nitrogen',
#             base_sensor_type='chemical',
#             sensor_type='soil_nitrogen',
#             value_type='float',
#             unit='mg/kg',
#             norm_range=[0.0, 1000.0],
#             physical_quantity='soil_nitrogen_content',
#             syntax_directives=[
#                 "Soil nitrogen sensor measures the available nitrogen content in the soil.",
#                 "Nitrogen is a key nutrient required for plant growth and protein synthesis.",
#                 "The sensor output informs fertilizer management and crop nutrition planning.",
#                 "It aids in evaluating soil fertility and guiding sustainable agricultural practices.",
#                 "Regular monitoring supports efficient nutrient use and improved crop yields."
#             ]
#         )
#         sensor_types.append(soil_nitrogen_sensor)

#         # Soil Phosphorus Sensor
#         soil_phosphorus_sensor = SensorType(
#             uid='soil_phosphorus',
#             base_sensor_type='chemical',
#             sensor_type='soil_phosphorus',
#             value_type='float',
#             unit='mg/kg',
#             norm_range=[0.0, 500.0],
#             physical_quantity='soil_phosphorus_content',
#             syntax_directives=[
#                 "Soil phosphorus sensor assesses the phosphorus concentration available to plants.",
#                 "Phosphorus is crucial for energy transfer and root development in crops.",
#                 "Sensor data helps determine soil fertility and phosphorus deficiency.",
#                 "It supports precise fertilizer application to optimize plant growth.",
#                 "Monitoring phosphorus levels assists in maintaining balanced soil nutrient profiles."
#             ]
#         )
#         sensor_types.append(soil_phosphorus_sensor)

#         # Soil Potassium Sensor
#         soil_potassium_sensor = SensorType(
#             uid='soil_potassium',
#             base_sensor_type='chemical',
#             sensor_type='soil_potassium',
#             value_type='float',
#             unit='mg/kg',
#             norm_range=[0.0, 1500.0],
#             physical_quantity='soil_potassium_content',
#             syntax_directives=[
#                 "Soil potassium sensor measures the concentration of potassium in the soil.",
#                 "Potassium is essential for water regulation and enzyme activation in plants.",
#                 "The sensor aids in assessing soil fertility and plant nutrient balance.",
#                 "It informs fertilizer strategies and helps improve crop stress resistance.",
#                 "Regular monitoring supports optimal nutrient management and sustainable yields."
#             ]
#         )
#         sensor_types.append(soil_potassium_sensor)
# # Assume sensor_types is an existing list collecting SensorType definitions

# # 1. Dew Point Sensor
#         dew_point_sensor = SensorType(
#             uid='dew_point',
#             base_sensor_type='temperature',
#             sensor_type='dew_point',
#             value_type='float',
#             unit='celsius',
#             norm_range=[-10.0, 30.0],
#             physical_quantity='dew_point_temperature',
#             syntax_directives=[
#                 "Dew point sensor measures the temperature at which air becomes saturated.",
#                 "It indicates the onset of condensation and is critical for forecasting fog and dew formation.",
#                 "Measurements are reported in degrees Celsius.",
#                 "Data supports precise prediction of microclimatic conditions in agricultural and environmental studies.",
#                 "Accurate dew point readings are essential for scheduling irrigation and frost protection."
#             ]
#         )
#         sensor_types.append(dew_point_sensor)

#         # 2. Vapor Pressure Sensor (vp)
#         vp_sensor = SensorType(
#             uid='vp',
#             base_sensor_type='gas',
#             sensor_type='vp',
#             value_type='float',
#             unit='kPa',
#             norm_range=[0.0, 4.0],
#             physical_quantity='vapor_pressure',
#             syntax_directives=[
#                 "Vapor pressure sensor measures the partial pressure exerted by water vapor in the air.",
#                 "It is vital for characterizing the moisture content in the atmosphere.",
#                 "Values are expressed in kilopascals (kPa).",
#                 "The sensor supports assessments of evaporative demand and plant transpiration.",
#                 "Accurate vapor pressure data enhances climate modeling and microclimate analysis."
#             ]
#         )
#         sensor_types.append(vp_sensor)

#         # 3. Vapor Pressure Deficit Sensor (vpd)
#         vpd_sensor = SensorType(
#             uid='vpd',
#             base_sensor_type='gas',
#             sensor_type='vapor_pressure_deficit',
#             value_type='float',
#             unit='kPa',
#             norm_range=[0.0, 5.0],
#             physical_quantity='vapor_pressure_deficit',
#             syntax_directives=[
#                 "Vapor pressure deficit (VPD) sensor measures the difference between saturation and actual vapor pressure.",
#                 "It is a key indicator of plant water stress and atmospheric dryness.",
#                 "Measurements are reported in kilopascals (kPa).",
#                 "VPD data aids in tailoring irrigation schedules and mitigating drought effects.",
#                 "Accurate VPD calculations are crucial for optimizing crop water management."
#             ]
#         )
#         sensor_types.append(vpd_sensor)

#         # 4. Sound Sensor
#         sound_sensor = SensorType(
#             uid='sound',
#             base_sensor_type='acoustic',
#             sensor_type='sound',
#             value_type='float',
#             unit='dB',
#             norm_range=[30.0, 120.0],
#             physical_quantity='ambient_sound_level',
#             syntax_directives=[
#                 "Sound sensor captures acoustic levels in the surrounding environment.",
#                 "It is essential for monitoring noise pollution in agricultural and urban areas.",
#                 "Measurements are provided in decibels (dB).",
#                 "The sensor aids in assessing the impact of environmental noise on both crops and livestock.",
#                 "Continuous acoustic monitoring supports improved environmental management."
#             ]
#         )
#         sensor_types.append(sound_sensor)

#         # 5. Flow Sensor
#         flow_sensor = SensorType(
#             uid='flow',
#             base_sensor_type='fluid',
#             sensor_type='flow',
#             value_type='float',
#             unit='L/min',
#             norm_range=[0.0, 500.0],
#             physical_quantity='fluid_flow_rate',
#             syntax_directives=[
#                 "Flow sensor measures the rate at which a fluid passes through a system.",
#                 "It is critical for managing irrigation systems and water distribution networks.",
#                 "Output is expressed in liters per minute (L/min).",
#                 "Real-time flow data supports efficient resource allocation and leak detection.",
#                 "Accurate monitoring improves pump management and operational safety."
#             ]
#         )
#         sensor_types.append(flow_sensor)

#         # 6. Level Sensor
#         level_sensor = SensorType(
#             uid='level',
#             base_sensor_type='mechanical',
#             sensor_type='level',
#             value_type='float',
#             unit='m',
#             norm_range=[0.0, 5.0],
#             physical_quantity='liquid_level',
#             syntax_directives=[
#                 "Level sensor measures the height of a liquid or material within a container or reservoir.",
#                 "It is essential for monitoring storage tanks and fluid inventory in systems.",
#                 "Measurements are reported in meters (m).",
#                 "Data assists in preventing overflows and optimizing refilling operations.",
#                 "Reliable level measurements enhance process safety and resource management."
#             ]
#         )
#         sensor_types.append(level_sensor)

#         # 7. Position Sensor (Directional)
#         position_sensor = SensorType(
#             uid='position',
#             base_sensor_type='positional',
#             sensor_type='position',
#             value_type='float',
#             unit='degrees',
#             norm_range=[0.0, 360.0],
#             physical_quantity='angular_position',
#             syntax_directives=[
#                 "Position sensor measures the angular orientation of an object.",
#                 "It encodes its output as a triplet: raw value, cosine, and sine of the angle (val_cos_sin).",
#                 "This directional encoding ensures smooth interpolation of angular data.",
#                 "Measurements are reported in degrees over a 0° to 360° range.",
#                 "Accurate positional data is vital for precise alignment and motion tracking."
#             ]
#         )
#         sensor_types.append(position_sensor)

#         # 8. Velocity Sensor (Directional)
#         velocity_sensor = SensorType(
#             uid='velocity',
#             base_sensor_type='positional',
#             sensor_type='velocity',
#             value_type='float',
#             unit='m/s',
#             norm_range=[0.0, 100.0],
#             physical_quantity='linear_velocity',
#             syntax_directives=[
#                 "Velocity sensor measures the speed and direction of movement.",
#                 "It outputs its directional data as a triplet: raw value, cosine, and sine of the angle (val_cos_sin).",
#                 "This encoding enables accurate representation of vector quantities.",
#                 "Measurements are provided in meters per second (m/s).",
#                 "Reliable velocity data supports dynamic process control and motion analysis."
#             ]
#         )
#         sensor_types.append(velocity_sensor)

#         # 9. Acceleration Sensor (Directional)
#         acceleration_sensor = SensorType(
#             uid='acceleration',
#             base_sensor_type='positional',
#             sensor_type='acceleration',
#             value_type='float',
#             unit='m/s²',
#             norm_range=[0.0, 20.0],
#             physical_quantity='linear_acceleration',
#             syntax_directives=[
#                 "Acceleration sensor measures the rate of change in velocity over time.",
#                 "It encodes directional information as a triplet: raw value, cosine, and sine of the angle (val_cos_sin).",
#                 "Such encoding is essential for capturing vector components of acceleration.",
#                 "Measurements are expressed in meters per second squared (m/s²).",
#                 "Accurate acceleration data is crucial for monitoring dynamic system behaviors and shocks."
#             ]
#         )
#         sensor_types.append(acceleration_sensor)

#         # 10. Direct Radiation Sensor
#         direct_radiation_sensor = SensorType(
#             uid='direct_radiation',
#             base_sensor_type='radiation',
#             sensor_type='direct_radiation',
#             value_type='float',
#             unit='W/m²',
#             norm_range=[0.0, 1400.0],
#             physical_quantity='direct_solar_radiation',
#             syntax_directives=[
#                 "Direct radiation sensor exclusively measures solar energy received directly from the sun.",
#                 "It excludes diffused and scattered components of sunlight.",
#                 "Values are reported in watts per square meter (W/m²).",
#                 "The sensor is pivotal for accurate solar irradiance and energy balance studies.",
#                 "Data supports precise agricultural and photovoltaic applications."
#             ]
#         )
#         sensor_types.append(direct_radiation_sensor)

#         # 11. Min Temperature Sensor
#         min_temperature_sensor = SensorType(
#             uid='min_temperature',
#             base_sensor_type='temperature',
#             sensor_type='min_temperature',
#             value_type='float',
#             unit='celsius',
#             norm_range=[-10.0, 40.0],
#             physical_quantity='daily_min_temperature',
#             syntax_directives=[
#                 "Min temperature sensor logs the lowest air temperature during a designated period.",
#                 "It helps identify potential cold stress and frost conditions.",
#                 "Measurements are provided in degrees Celsius.",
#                 "Accurate readings support frost forecasting and risk mitigation in agriculture.",
#                 "Continuous monitoring aids in developing adaptive crop protection strategies."
#             ]
#         )
#         sensor_types.append(min_temperature_sensor)

#         # 12. Wet Temperature Sensor
#         wet_temperature_sensor = SensorType(
#             uid='wet_temperature',
#             base_sensor_type='temperature',
#             sensor_type='wet_temperature',
#             value_type='float',
#             unit='celsius',
#             norm_range=[-10.0, 35.0],
#             physical_quantity='wet_temperature',
#             syntax_directives=[
#                 "Wet temperature sensor measures the temperature of a wetted thermometer bulb.",
#                 "It is key for deriving the wet-bulb temperature critical in humidity and evaporation studies.",
#                 "Readings are expressed in degrees Celsius.",
#                 "The sensor supports calculations for evaporation rates and cooling processes.",
#                 "Accurate wet temperature data assists in effective climate control and irrigation planning."
#             ]
#         )
#         sensor_types.append(wet_temperature_sensor)

#         # 13. Wind Direction Sensor (Directional)
#         wind_direction_sensor = SensorType(
#             uid='wind_direction',
#             base_sensor_type='wind',
#             sensor_type='wind_direction',
#             value_type='float',
#             unit='degrees',
#             norm_range=[0.0, 360.0],
#             physical_quantity='wind_direction',
#             syntax_directives=[
#                 "Wind direction sensor determines the origin of the wind.",
#                 "It encodes its output as a triplet: raw value, cosine, and sine of the angle (val_cos_sin).",
#                 "This method ensures smooth directional interpolation.",
#                 "Measurements span the full 0° to 360° range.",
#                 "Data supports weather forecasting and precise agricultural planning."
#             ]
#         )
#         sensor_types.append(wind_direction_sensor)

#         # 14. Gust Wind Direction Sensor (Directional)
#         gust_wind_direction_sensor = SensorType(
#             uid='gust_wind_direction',
#             base_sensor_type='wind',
#             sensor_type='gust_wind_direction',
#             value_type='float',
#             unit='degrees',
#             norm_range=[0.0, 360.0],
#             physical_quantity='gust_wind_direction',
#             syntax_directives=[
#                 "Gust wind direction sensor measures the direction of transient wind gusts.",
#                 "Its output is encoded as a triplet: raw value, cosine, and sine of the angle (val_cos_sin).",
#                 "This detailed encoding helps capture rapid directional changes.",
#                 "The sensor covers the entire circular range from 0° to 360°.",
#                 "Accurate gust direction data aids in dynamic weather and safety assessments."
#             ]
#         )
#         sensor_types.append(gust_wind_direction_sensor)

#         # 15. Max 1-Minute Wind Speed Sensor
#         max_1min_wind_speed_sensor = SensorType(
#             uid='max_1min_wind_speed',
#             base_sensor_type='wind',
#             sensor_type='max_1min_wind_speed',
#             value_type='float',
#             unit='m/s',
#             norm_range=[0.0, 60.0],
#             physical_quantity='max_1min_wind_speed',
#             syntax_directives=[
#                 "Max 1-minute wind speed sensor records the highest wind speed within a one-minute interval.",
#                 "It is critical for capturing short-term gust extremes.",
#                 "Measurements are provided in meters per second (m/s).",
#                 "Data assists in immediate weather and safety evaluations.",
#                 "Accurate peak wind readings support dynamic load assessments on structures."
#             ]
#         )
#         sensor_types.append(max_1min_wind_speed_sensor)

#         # 16. Max 10-Minute Wind Speed Sensor
#         max_10min_wind_speed_sensor = SensorType(
#             uid='max_10min_wind_speed',
#             base_sensor_type='wind',
#             sensor_type='max_10min_wind_speed',
#             value_type='float',
#             unit='m/s',
#             norm_range=[0.0, 60.0],
#             physical_quantity='max_10min_wind_speed',
#             syntax_directives=[
#                 "Max 10-minute wind speed sensor logs the peak wind speed over a ten-minute period.",
#                 "It provides insights into sustained wind conditions.",
#                 "Measurements are reported in meters per second (m/s).",
#                 "This sensor supports regional weather forecasting and structural analysis.",
#                 "Extended period monitoring facilitates risk management under prolonged wind events."
#             ]
#         )
#         sensor_types.append(max_10min_wind_speed_sensor)

#         # 17. Time Ending Max 10-Minute Wind Speed Sensor
#         time_ending_max_10min_wind_speed_sensor = SensorType(
#             uid='time_ending_max_10min_wind_speed',
#             base_sensor_type='wind',
#             sensor_type='time_ending_max_10min_wind_speed',
#             value_type='float',
#             unit='seconds',
#             norm_range=[0.0, 86400.0],  # example: within a day (0 to 86400 seconds)
#             physical_quantity='time_ending_max_10min_wind_speed',
#             syntax_directives=[
#                 "This sensor records the time at which the 10-minute maximum wind speed occurred.",
#                 "It provides temporal context critical for wind event analysis.",
#                 "Measurements are reported in seconds relative to a reference time.",
#                 "Data is vital for correlating wind speed peaks with weather events.",
#                 "Accurate timing information supports detailed meteorological assessments."
#             ]
#         )
#         sensor_types.append(time_ending_max_10min_wind_speed_sensor)

#         # 18. Gust Wind Speed Sensor
#         gust_wind_speed_sensor = SensorType(
#             uid='gust_wind_speed',
#             base_sensor_type='wind',
#             sensor_type='gust_wind_speed',
#             value_type='float',
#             unit='m/s',
#             norm_range=[0.0, 60.0],
#             physical_quantity='gust_wind_speed',
#             syntax_directives=[
#                 "Gust wind speed sensor measures the instantaneous speed of wind gusts.",
#                 "It is essential for capturing brief spikes in wind velocity.",
#                 "Measurements are provided in meters per second (m/s).",
#                 "The sensor supports real-time hazard assessments and dynamic modeling.",
#                 "Accurate gust speed data is critical for structural safety and weather forecasting."
#             ]
#         )
#         sensor_types.append(gust_wind_speed_sensor)

#         # 19. Standard Deviation Wind Direction Sensor (Directional)
#         std_wind_direction_sensor = SensorType(
#             uid='std_wind_direction',
#             base_sensor_type='wind',
#             sensor_type='std_wind_direction',
#             value_type='float',
#             unit='degrees',
#             norm_range=[0.0, 180.0],
#             physical_quantity='std_wind_direction',
#             syntax_directives=[
#                 "This sensor computes the standard deviation of wind direction measurements.",
#                 "It encodes the variability as a triplet: raw value, cosine, and sine of the computed angle (val_cos_sin).",
#                 "This directional encoding helps quantify turbulence and spread in wind patterns.",
#                 "Measurements aid in evaluating the consistency of wind direction over time.",
#                 "Accurate directional variability is critical for risk assessments in wind-sensitive operations."
#             ]
#         )
#         sensor_types.append(std_wind_direction_sensor)

#         # 20. Rainfall Sensor
#         rainfall_sensor = SensorType(
#             uid='rainfall',
#             base_sensor_type='hydrological',
#             sensor_type='rainfall',
#             value_type='float',
#             unit='mm',
#             norm_range=[0.0, 500.0],
#             physical_quantity='rainfall',
#             syntax_directives=[
#                 "Rainfall sensor measures the accumulated precipitation over a set period.",
#                 "It is essential for hydrological assessments and irrigation planning.",
#                 "Measurements are reported in millimeters (mm).",
#                 "Data supports flood risk evaluation and water resource management.",
#                 "Accurate rainfall monitoring is key to sustainable agricultural practices."
#             ]
#         )
#         sensor_types.append(rainfall_sensor)

#         # 21. Soil pH Sensor (Soil-Specific)
#         soil_ph_sensor = SensorType(
#             uid='soil_ph',
#             base_sensor_type='chemical',
#             sensor_type='soil_ph',
#             value_type='float',
#             unit='pH',
#             norm_range=[3.5, 9.0],
#             physical_quantity='soil_ph',
#             syntax_directives=[
#                 "Soil pH sensor specifically measures the acidity or alkalinity of soil.",
#                 "It is critical for assessing soil fertility and nutrient availability.",
#                 "Values are unitless and reported on a logarithmic scale.",
#                 "Accurate soil pH data supports precision fertilization and amendment planning.",
#                 "Monitoring soil pH helps optimize crop production and sustainable land management."
#             ]
#         )
#         sensor_types.append(soil_ph_sensor)