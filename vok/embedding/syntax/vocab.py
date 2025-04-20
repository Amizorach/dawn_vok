class DVocab:
    @classmethod
    def get_syntax_vocab(cls):
        return {
            "unit_type": ["sensor", "actuator", "controller", "processor", "memory", "storage", "network", "display", "input", "output"],
            "sensor": ["temperature", "humidity",
                       "dew point", "vp", "vpd", "light", "sound", "pressure", "flow", "level", "position", "velocity", "acceleration"],
        "sensor_sub_type": ["temperature", "humidity", "radiation", "soil_moisture", "air_pressure",
                            'diffused radiation', 'global radiation',
        'direct radiation', 'humidity', 'temperature', 'max temperature',
        'min temperature', 'wet temperature', 'wind direction',
        'gust wind direction', 'wind speed', 'max 1 minute wind speed',
        'max 10 minute wind speed', 'time ending max 10 minute wind speed',
        'gust wind speed', 'standard deviation wind direction', 'rainfall'
        "wind speed", "co2", "ph", "ec", "light intensity", "rainfall", "wind direction", "wind speed",
        "solar radiation", "soil temperature", "soil moisture", "soil ph", "soil ec", "soil organic matter", "soil nitrogen", "soil phosphorus", "soil potassium",
        ],
        # "manufacturers": ["Netafim", "Bosch", "Parrot", "AgriTech", "SensorCo", "WeatherCo", "PlantCo", "SoilCo", "WaterCo", "LightCo", "Grofit", "other"],
        "locations": ["greenhouse", 'outdoor', 'indoor', 'field', 'orchard', 'farm', 'lab', 'office', 'home', 'other'],
        "value_types": ["float", "int", "str", "time", "date", "datetime", "duration", "enum", "boolean", "image", "audio", "video", "document", "spreadsheet", "presentation", "email", "other"],
        "value_type_units": ["celcius", "fahrenheit", "kelvin", "pascal", "hPa (hectopascal)", "kPa (kilopascal)", "MPa (megapascal)", "GPa (gigapascal)", "m/s (miles per second)", "km/h (kilometers per hour)", "mph (miles per hour)", "kph (kilometers per hour)",
                        "km/s (kilometers per second)", "m/s (meters per second)", "km/h (kilometers per hour)", "mph (miles per hour)", "kph (kilometers per hour)", "km/s (kilometers per second)"],


        "aggregation_functions": ["min", "max", "mean", "median", "sum", "count", "std", "var", "other"],
        "formulation_functions": ['svd', 'pca', 'other']   
        }




    def get_discrete_vocab(self):
        return {
        "manufacturers": ["Netafim", "Bosch", "Parrot", "AgriTech", "SensorCo", "WeatherCo", "PlantCo", "SoilCo", "WaterCo", "LightCo", "Grofit", "other"],
        }
    
    
    # def get_vocab(self, key):
    #     return self.vocab_dict[key]
    
    # def get_vocab_size(self, key):
    #     return len(self.vocab_dict[key])


   