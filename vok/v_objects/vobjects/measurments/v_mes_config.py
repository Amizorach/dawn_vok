   # definitions for units used across sensors
class MeasurementConfig:

    measurment_unit_info = {
        'celsius': {
            'symbol': '°C',
            'syntax_directives': [
                "Metric temperature unit where 0°C is water's freezing point.",
                'Widely used in scientific and environmental measurements.',
                'Directly convertible to Kelvin and Fahrenheit scales.'
            ]
        },
        'fahrenheit': {
            'symbol': '°F',
            'syntax_directives': [
                "Imperial temperature unit where 32°F is water's freezing point.",
                'Commonly used in the United States and a few other countries.',
                'Directly convertible to Celsius and Kelvin scales.'
            ]
        },
        'kelvin': {
            'symbol': 'K',
            'syntax_directives': [
                'SI base unit of temperature where 0 K represents absolute zero.',
                'Widely used in scientific research.',
                'Conversion formula: K = °C + 273.15.'
            ]
        },
        'percent': {
            'symbol': '%',
            'syntax_directives': [
                'Represents a ratio expressed as parts per hundred.',
                'Commonly used for relative humidity and concentration measurements.',
                'Dimensionless unit.'
            ]
        },
        'pascal': {
            'symbol': 'Pa',
            'syntax_directives': [
                'SI unit of pressure equal to one newton per square meter.',
                'Widely used in engineering and meteorology.',
                'Foundation for derived units like hectopascal and bar.'
            ]
        },
        'kilopascal': {
            'symbol': 'kPa',
            'syntax_directives': [
                'Derived unit of pressure equal to 1,000 pascals.',
                'Commonly used in meteorology and engineering for atmospheric pressure.',
                '1 kPa = 0.01 bar.'
            ]
        },
        'm_s': {
            'symbol': 'm_s',
            'syntax_directives': [
                'SI unit for speed or velocity.',
                'Used for wind speed and flow measurements.',
                'Expresses distance traveled per unit time.'
            ]
        },
        'degrees': {
            'symbol': '°',
            'syntax_directives': [
                'Angular unit for directional measurements.',
                'Used for wind direction and orientation.',
                'Ranges from 0° (north) clockwise to 360°.'
            ]
        },
        'hectopascal': {
            'symbol': 'hPa',
            'syntax_directives': [
                    'Equal to 100Pa; standard unit for atmospheric pressure.',
                    'Common in aviation and synoptic weather maps.',
                    '1hPa=0.1kPa=0.02953inHg.'
                ]
            },
            'millimeter': {
                'symbol': 'mm',
                'syntax_directives': [
                    'Metric length unit; used for accumulated rainfall depth.',
                    '1mm of rain corresponds to 1Lm⁻² of water.',
                    'Directly convertible to inches (1mm≈0.03937 in).'
                ]
            },
            'meter': {
                'symbol': 'm',
                'syntax_directives': [
                    'SI base unit of length.',
                    'Used for visibility range, cloudceiling height, and sensor elevation.',
                    'Foundation for derived units such as kilometers and millimeters.'
                ]
            },
            'okta': {
                'symbol': '/8',
                'syntax_directives': [
                    'Fractional sky cover where 0=clear and 8=overcast.',
                    'Standard in WMO cloud observation practice.',
                    'Dimensionless but treated as an integer scale.'
                ]
            },
            'none': {
                'symbol': '',
                'syntax_directives': [
                    'Placeholder for dimensionless, coded, or textual quantities.',
                    'Used when the recorded value is categorical or free text (e.g., METAR string).',
                    'No numerical unit conversion applies.'
                ]
            },
        }
