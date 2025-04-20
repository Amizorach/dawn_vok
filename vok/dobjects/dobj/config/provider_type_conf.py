
class ProviderTypeConfig:
    type_info = [
        {
            "_id": "provider_ims",              # Updated: "provider_" + original "ims"
            "uid": "ims",                       # New field with original _id value
            "name": "Israel Meteorological Service (IMS)",
            "url": "https://ims.gov.il/en",
            "api_url": "https://ims.gov.il/en/ObservationDataAPI",
            "meta_url": "https://ims.gov.il/en/stations",
            "syntax_directives": [
                "The Israel Meteorological Service (IMS) operates a national network of meteorological stations throughout Israel.",
                "This network includes both automated weather stations and hundreds of dedicated volunteer rainfall observers.",
                "IMS stations continuously measure key meteorological parameters such as temperature, rainfall, wind speed, and humidity.",
                "Observations from the IMS station network are fundamental for weather forecasting, climate analysis, and research in Israel.",
                "Meteorological data gathered from IMS stations is systematically archived and made available for various applications."
            ],
        },
        {
            "_id": "provider_synoptic_concept", # Updated
            "uid": "synoptic_concept",          # New
            "name": "Synoptic Weather Station Concept",
            "url": "https://wmo.int/",
            "api_url": "https://community.wmo.int/en/activity-areas/wis/wis2-implementation",
            "meta_url": "https://library.wmo.int/en/resources/meteoterm/concept/Synoptic%20station",
            "syntax_directives": [
                "Synoptic weather stations operate under strict international guidelines for instrument placement and data accuracy.",
                "These facilities transmit meteorological observations at standardized times using internationally agreed-upon codes.",
                "Data gathered from this type of station network is essential input for operational numerical weather prediction systems.",
                "They form part of a coordinated, wide-area observation network designed for large-scale atmospheric monitoring.",
                "Observations undergo rigorous quality control procedures and contribute to official national and global climate archives."
            ],
        },
        {
            "_id": "provider_noaa_nws_api",      # Updated
            "uid": "noaa_nws_api",              # New
            "name": "NOAA / National Weather Service API (api.weather.gov)",
            "url": "https://www.weather.gov/",
            "api_url": "https://www.weather.gov/documentation/services-web-api",
            "meta_url": "https://weather-gov.github.io/api/general-faqs",
            "syntax_directives": [
                "This public API delivers official forecasts and recent weather observations primarily for locations within the United States.",
                "It requires application identification via User-Agent for accessing standardized US weather service data streams.",
                "Access involves querying by geographic coordinates which are then mapped to a specific forecast grid and associated reporting stations.",
                "The service provides real-time alerts, hourly forecasts, and current conditions directly from government meteorological sources.",
                "While offering recent instrumental readings, extensive historical climate records necessitate querying separate archival systems from the same agency."
            ],
            "value_type": "float",
        },
        {
            "_id": "provider_noaa_ndbc",         # Updated
            "uid": "noaa_ndbc",                 # New
            "location": None,
            "name": "NOAA National Data Buoy Center (NDBC)",
            "url": "https://www.ndbc.noaa.gov/",
            "api_url": "https://www.ndbc.noaa.gov/faq/realtime.shtml",
            "meta_url": "https://www.ndbc.noaa.gov/docs.shtml",
            "system_uid": None,                 # Remains None
            "syntax_directives": [
                "This governmental center provides real-time meteorological and oceanographic data from moored buoys and coastal stations.",
                "Its focus is on marine conditions, including wind, wave height, and water temperature, primarily in US waters.",
                "Recent observational data is readily available online through station lists and map interfaces.",
                "Accessing deep historical archives often involves navigating specific data directories or web-based query tools.",
                "Data from this network is crucial for marine forecasting, research, and tsunami monitoring efforts."
            ],
            "value_type": "float",
            "updated_at":  None,
        },
        {
            "_id": "provider_met_norway",       # Updated
            "uid": "met_norway",                # New
            "location": None,
            "name": "MET Norway APIs",
            "url": "https://www.met.no/en",
            "api_url": "https://api.met.no/",
            "meta_url": "https://api.met.no/weatherapi/frost/1.0/documentation",
            "system_uid": None,                 # Remains None
            "syntax_directives": [
                "This European national meteorological service offers free public access to various weather datasets via API.",
                "Observational data from a wide network, including global coverage points, is accessible through its dedicated 'Frost' REST API endpoint.",
                "The service mandates application identification through a User-Agent header for usage tracking and communication.",
                "Specific APIs are provided for aviation standard reports like METAR and TAF, alongside regional forecasts.",
                "Data usage is governed by a Creative Commons license, encouraging broad application including commercial use."
            ],
            "value_type": "float",
            "updated_at":  None,
        },
        {
            "_id": "provider_synoptic_data",     # Updated
            "uid": "synoptic_data",             # New
            "location": None,
            "name": "Synoptic Data (Company)",
            "url": "https://synopticdata.com/",
            "api_url": "https://synopticdata.com/weatherapi/",
            "meta_url": "https://developers.synopticdata.com/mesonet/v2/stations/metadata/",
            "system_uid": None,                 # Remains None
            "syntax_directives": [
                "This commercial service aggregates weather observations from an extremely large number (tens of thousands) of global public and private stations.",
                "It specializes in providing access to both real-time and extensive historical instrumental data via a flexible RESTful API.",
                "Access requires user registration and the use of authentication tokens for API requests.",
                "The platform emphasizes delivering raw and quality-controlled observational data from diverse worldwide networks.",
                "Data can be retrieved in various formats like JSON and CSV, tailored for professional and analytical use cases."
            ],
            "value_type": "float",
            "updated_at":  None,
        },
        {
            "_id": "provider_weatherbit",       # Updated
            "uid": "weatherbit",                # New
            "location": None,
            "name": "Weatherbit",
            "url": "https://www.weatherbit.io/",
            "api_url": "https://www.weatherbit.io/api",
            "meta_url": "https://www.weatherbit.io/api",
            "system_uid": None,                 # Remains None
            "syntax_directives": [
                "This commercial API provider offers global weather data, integrating observations from over 50,000 stations with other sources.",
                "It emphasizes machine learning techniques to enhance forecast accuracy derived from multiple inputs, including instrumental readings.",
                "Significant historical depth (claiming 20+ years) for observational and derived data is accessible via dedicated API endpoints.",
                "Access requires an API key, with various subscription tiers available, including a free starting point.",
                "Data includes standard meteorological variables plus specialized parameters like air quality and agricultural metrics."
            ],
            "value_type": "float",
            "updated_at":  None,
        },
        {
            "_id": "provider_dtn",              # Updated
            "uid": "dtn",                       # New
            "location": None,
            "name": "DTN",
            "url": "https://www.dtn.com/",
            "api_url": "https://api.weather.mg/",
            "meta_url": "https://api.weather.mg/",
            "system_uid": None,                 # Remains None
            "syntax_directives": [
                "This provider offers weather data APIs targeted towards business and specific industry applications.",
                "A dedicated Point Observation API endpoint allows retrieval of instrumental weather data for specified locations or stations.",
                "Access to their services is typically managed through contracts or trials, requiring client credentials and access tokens.",
                "They emphasize reliable, scalable data delivery integrating global stations, radar, and satellite inputs for professional insights.",
                "The focus is often on delivering actionable weather intelligence tailored to operational business needs."
            ],
            "value_type": "float",
            "updated_at":  None,
        },
        {
            "_id": "provider_wu_ibm_twc",       # Updated
            "uid": "wu_ibm_twc",                # New
            "location": None,
            "name": "Weather Underground / IBM / The Weather Company",
            "url": "https://www.ibm.com/weather",
            "api_url": "https://www.weathercompany.com/weather-data-apis/",
            "meta_url": "https://www.wunderground.com/pws/overview",
            "system_uid": None,                 # Remains None
            "syntax_directives": [
                "This commercial entity, part of a major technology corporation, leverages a vast global network of personal weather stations (PWS).",
                "It provides API access to hyper-local current conditions and historical data contributed by hundreds of thousands of citizen-operated sensors.",
                "Access typically requires an API key obtained through their platform, often integrated with broader enterprise data solutions.",
                "The service uniquely blends official data sources with its extensive PWS network for granular weather insights.",
                "Historical instrumental data, particularly from the PWS network, is a key offering for trend analysis and research."
            ],
            "value_type": "float",
            "updated_at":  None,
        },
        {
            "_id": "provider_noaa_aviation",     # Updated
            "uid": "noaa_aviation",             # New
            "location": None,
            "name": "NOAA Aviation Weather Center API",
            "url": "https://aviationweather.gov/",
            "api_url": "https://aviationweather.gov/data/api/",
            "meta_url": "https://aviationweather.gov/data/api/",
            "system_uid": None,                 # Remains None
            "syntax_directives": [
                "This specialized government API provides official meteorological information specifically tailored for the aviation industry.",
                "It delivers standardized METAR (current observation) and TAF (terminal forecast) reports from airports globally.",
                "The focus is on operational, real-time, and recent data crucial for flight planning and safety.",
                "Access is typically public but designed for automated systems involved in aviation operations.",
                "While providing critical current reports, deep historical aviation weather records are usually archived elsewhere by the parent agency."
            ],
            "value_type": "float",
            "updated_at":  None,
        },
        {
            "_id": "provider_openweathermap",    # Updated
            "uid": "openweathermap",            # New
            "location": None,
            "name": "OpenWeatherMap",
            "url": "https://openweathermap.org/",
            "api_url": "https://openweathermap.org/api",
            "meta_url": "https://openweathermap.org/stations",
            "system_uid": None,                 # Remains None
            "syntax_directives": [
                "This widely-used commercial service provides global weather data via API, utilized by a large developer community.",
                "It aggregates data from numerous sources, including a large network of weather stations and user-contributed station data.",
                "The API offers current conditions, forecasts, and extensive historical weather data access, often with a free usage tier.",
                "It provides specific API endpoints for users to register and manage their own personal weather stations.",
                "Data is delivered in common formats like JSON, supporting diverse applications from hobby projects to commercial products."
            ],
            "value_type": "float",
            "updated_at":  None,
        },
        {
            "_id": "provider_noaa_ncei",        # Updated
            "uid": "noaa_ncei",                 # New
            "location": None,
            "name": "NOAA / NCEI (National Centers for Environmental Information)",
            "url": "https://www.ncei.noaa.gov/",
            "api_url": "https://www.ncei.noaa.gov/access",
            "meta_url": "https://www.ncei.noaa.gov/products/land-based-station",
            "system_uid": None,                 # Remains None
            "syntax_directives": [
                "This organization serves as the primary US national archive for environmental and climate data.",
                "It provides access to deep historical meteorological records, including decades of daily and monthly data from official weather stations.",
                "Access to these extensive historical datasets is often facilitated through specialized web services, APIs (requiring tokens), or data discovery tools.",
                "The focus is on providing long-term, quality-controlled observational data for climate research, analysis, and official records.",
                "Data formats and access methods vary depending on the specific dataset (e.g., GHCN-Daily, GSOM) being queried."
            ],
            "value_type": "float",
            "updated_at":  None,
        }
    ]
