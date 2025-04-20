import ee
from datetime import datetime

class GELocationUtils:
    soil_property_info_map = {
        "soil_ph": {
            "name": "soil pH",
            "units": "pH units",
            "syntax_directives": [
                "This location exhibits a soil pH level indicative of {value}, which affects nutrient availability.",
                "The soil is considered {acidity_class} based on a pH of {value}.",
                "pH influences microbial activity and plant root behavior in this region."
            ]
        },
        "soil_ocd": {
            "name": "organic carbon density",
            "units": "g/kg",
            "syntax_directives": [
                "The topsoil contains {value} grams of organic carbon per kilogram.",
                "Higher organic carbon levels enhance soil fertility and structure.",
                "Organic content reflects biological activity and potential for nutrient retention."
            ]
        },
        "soil_bdod": {
            "name": "bulk density",
            "units": "kg/m³",
            "syntax_directives": [
                "The bulk density of the topsoil is {value} kg/m³, influencing compaction and porosity.",
                "Lower bulk density indicates better root penetration and air exchange.",
                "Soil density affects water retention and mechanical resistance."
            ]
        },
        "soil_clay": {
            "name": "clay percentage",
            "units": "%",
            "syntax_directives": [
                "Clay makes up {value}% of the soil texture at this location.",
                "High clay content increases water-holding capacity but can reduce aeration.",
                "Clayey soils tend to swell and shrink with moisture fluctuations."
            ]
        },
        "soil_sand": {
            "name": "sand percentage",
            "units": "%",
            "syntax_directives": [
                "Sand constitutes {value}% of the soil, promoting drainage and aeration.",
                "High sand levels are typical of well-draining, low-retention soils.",
                "Soils with more sand warm quickly and dry out faster."
            ]
        },
        "soil_silt": {
            "name": "silt percentage",
            "units": "%",
            "syntax_directives": [
                "Silt represents {value}% of the soil's texture profile.",
                "Silt improves soil smoothness and water-holding without compaction.",
                "Soils rich in silt are fertile but may be erosion-prone."
            ]
        }
    }
    terrain_property_info_map = {
        "elevation": {
            "name": "elevation",
            "units": "meters above sea level",
            "syntax_directives": [
                "The location sits at an elevation of approximately {value} meters above sea level.",
                "Elevated terrain at {value} meters may influence local climate and vegetation.",
                "Topographic elevation is {value} m, which affects drainage and atmospheric conditions."
            ]
        },
        "slope": {
            "name": "slope",
            "units": "degrees",
            "syntax_directives": [
                "The slope at this point is {value}°, influencing water runoff and erosion.",
                "Terrain inclination is measured at {value} degrees.",
                "Sloped surfaces can affect soil stability and vegetation patterns."
            ]
        },
        "aspect": {
            "name": "aspect",
            "units": "degrees from north",
            "syntax_directives": [
                "The aspect is {value}°, indicating the direction the slope faces.",
                "This terrain faces {direction}, influencing sunlight exposure and microclimates.",
                "An aspect of {value} degrees modifies evapotranspiration and growth patterns."
            ]
        },
        "land_cover": {
            "name": "land cover",
            "syntax_directives": [
                "The dominant land cover type is {label}, suggesting characteristic vegetation and surface properties.",
                "Surface classification identifies this region as primarily {label}.",
                "Land cover at this point is categorized as {label}."
            ]
        },
        "ndvi": {
            "name": "Normalized Difference Vegetation Index",
            "units": "ratio (-1 to 1)",
            "syntax_directives": [
                "The NDVI value is {value}, indicating {vegetation_class} vegetation density.",
                "Vegetation greenness is measured at NDVI = {value}, which reflects {vegetation_class} photosynthetic activity.",
                "An NDVI of {value} suggests {vegetation_class} canopy coverage."
            ]
        }
    }

    @classmethod
    def decode_land_cover_label(cls, label):
        land_cover_info_map = {
            0: {
                'key': 'water',
                'gee_id': 0,
                'name': 'water',
                'syntax_directives': [
                    'This region is dominated by open water bodies such as lakes, rivers, or oceans.',
                    'The surface has low vegetation and may reflect high levels of sunlight.',
                    'Aquatic environments can affect local climate and evaporation.'
                ]
            },
            1: {
                'key': 'trees',
                'gee_id': 1,
                'name': 'trees',
                'syntax_directives': [
                    'This area is densely covered by tall, woody vegetation.',
                    'Forested zones often support biodiversity and carbon sequestration.',
                    'Canopy cover reduces surface albedo and influences microclimate.'
                ]
            },
            2: {
                'key': 'grass',
                'gee_id': 2,
                'name': 'grass',
                'syntax_directives': [
                    'This region is covered with short, photosynthetically active vegetation.',
                    'The area likely supports grassland ecosystems and open grazing fields.',
                    'Land surface is predominantly green during the growing season.'
                ]
            },
            3: {
                'key': 'flooded vegetation',
                'gee_id': 3,
                'name': 'flooded vegetation',
                'syntax_directives': [
                    'This area contains vegetation that is seasonally or permanently submerged.',
                    'Wetland species dominate this mixed aquatic-terrestrial environment.',
                    'These regions are crucial for water filtration and biodiversity.'
                ]
            },
            4: {
                'key': 'crops',
                'gee_id': 4,
                'name': 'crops',
                'syntax_directives': [
                    'This region is used for agricultural production of seasonal or perennial crops.',
                    'Vegetation patterns vary with planting and harvesting cycles.',
                    'Surface properties are managed and altered by human activity.'
                ]
            },
            5: {
                'key': 'shrub_and_scrub',
                'gee_id': 5,
                'name': 'shrub and scrub',
                'syntax_directives': [
                    'The area is dominated by low woody vegetation with sparse canopy.',
                    'This land cover type is typical in semi-arid and transitional zones.',
                    'Soil exposure is moderate, and vegetation is drought-tolerant.'
                ]
            },
            6: {
                'key': 'built_up_area',
                'gee_id': 6,
                'name': 'built up area',
                'syntax_directives': [
                    'This location is characterized by man-made surfaces such as buildings and roads.',
                    'Urban and suburban structures dominate land use.',
                    'High impervious surface area impacts water runoff and temperature.'
                ]
            },
            7: {
                'name': 'bare ground',
                'syntax_directives': [
                    'This region has exposed soil, rock, or non-vegetated surfaces.',
                    'Vegetation cover is minimal or absent.',
                    'May indicate erosion-prone or recently disturbed land.'
                ]
            },
            8: {
                'key': 'snow_and_ice',
                'gee_id': 8,
                'name': 'snow and ice',
                'syntax_directives': [
                    'This area is covered by snow or glacial ice, either seasonally or permanently.',
                    'High reflectance contributes to low surface energy absorption.',
                    'Temperature-sensitive and typically found in high latitudes or elevations.'
                ]
            },
            9: {
                'key': 'clouds',
                'gee_id': 9,
                'name': 'clouds',
                'syntax_directives': [
                    'The satellite view is obscured by atmospheric clouds.',
                    'Surface observation may be invalid or incomplete.',
                    'No reliable surface land cover detected due to cloud interference.'
                ]
            }
        }

        return land_cover_info_map.get(label, {"name": "unknown", "syntax_directives": []})

    def __init__(self, lat, lon, project_id="ee-amizorach"):
        """
        Initialize the GGELocation with a latitude, longitude, and a Google Earth Engine project ID.
        
        Args:
            lat (float): Latitude in decimal degrees.
            lon (float): Longitude in decimal degrees.
            project_id (str): The Google Earth Engine project ID. Default is "ee-amizorach".
        """
        self.lat = lat
        self.lon = lon
        self.project_id = project_id
        # Initialize Earth Engine with the specified project ID.
        ee.Initialize(project=project_id)
        # Create a point geometry (Earth Engine expects [lon, lat]).
        self.point = ee.Geometry.Point(lon, lat)
    
    def get_elevation_and_terrain(self):
        """
        Retrieve elevation, slope, and aspect information in one combined query.
        
        Returns:
            dict: A dictionary containing 'elevation' (meters), 'slope' (degrees),
                  and 'aspect' (degrees).
        """
        srtm = ee.Image("USGS/SRTMGL1_003")
        
        # Sample elevation at the point (using a 30m resolution).
        sample = srtm.sample(self.point, 30).first()
        elevation = sample.get('elevation').getInfo()
        
        # Compute terrain products (slope and aspect) from the SRTM data.
        terrain = ee.Terrain.products(srtm)
        sample_terrain = terrain.sample(self.point, 30).first()
        slope = sample_terrain.get('slope').getInfo()
        aspect = sample_terrain.get('aspect').getInfo()
        
        return {
            "elevation": elevation,
            "slope": slope,
            "aspect": aspect
        }
    def get_land_cover_dynamic_world(self, date=None):
        """
        Retrieve daily land cover data using the Dynamic World product.
        This method filters the Dynamic World image collection for the given day,
        composites the available images to produce a daily composite, and then samples
        the 'label' band at the provided location.
        
        Args:
            date (datetime, optional): The date for which to retrieve land cover data.
                                    If None, the current UTC date is used.
        
        Returns:
            int or None: The land cover classification label from Dynamic World,
                        or None if no valid sample is available.
        """
        if date is None:
            date = datetime(2025, 1, 1)
        # Format the date as YYYY-MM-DD.
        start_date = ee.Date(date.strftime('%Y-%m-%d'))
        # Use a one-day window.
        end_date = start_date.advance(100, 'day')
        
        dynamic_world = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1") \
                            .filterDate(start_date, end_date) \
                            .filterBounds(self.point)
        
        # Composite available images from the day.
        daily_image = dynamic_world.mosaic()
        
        # Check if the composite image contains any bands.
        try:
            bands = daily_image.bandNames().getInfo()
        except Exception as e:
            print("Error retrieving band names:", e)
            return None

        if not bands:
            print("Daily composite image has no bands; likely no data available for this date and location.")
            return None

        # Sample the 'label' band at the point using a scale of 10m (native resolution).
        sample_result = daily_image.select("label").sample(self.point, 10).first()
        
        if sample_result is None:
            print("No sample result from the daily composite for the given point.")
            return None
        
        try:
            land_cover = sample_result.get('label').getInfo()
        except Exception as e:
            print("Could not retrieve land cover label from the sample:", e)
            return None
        
        return land_cover
    def get_ndvi_sentinel2(self, date=None):
        """
        Computes NDVI using Sentinel-2 surface reflectance data (10m resolution).
        
        Returns:
            float or None: NDVI value or None if not found.
        """
        if date is None:
            date = datetime(2025, 1, 1)

        start_date = ee.Date(date.strftime('%Y-%m-%d'))
        end_date = start_date.advance(5, 'day')

        s2_sr = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterBounds(self.point) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))  # relaxed to 40%

        def add_ndvi(image):
            return image.addBands(image.normalizedDifference(['B8', 'B4']).rename('NDVI'))

        s2_ndvi = s2_sr.map(add_ndvi)

        # Make sure collection is not empty
        size = s2_ndvi.size().getInfo()
        if size == 0:
            print("Sentinel-2: No images found in this date range.")
            return None

        image = s2_ndvi.first()
        if image is None:
            print("Sentinel-2: No image available.")
            return None

        try:
            sample = image.select('NDVI').sample(self.point, 10).first()
            if sample:
                return sample.get('NDVI').getInfo()
        except Exception as e:
            print("Sentinel-2 NDVI error:", e)

        return None

    def get_ndvi_modis(self, date=None):
        """
        Retrieve NDVI value at the point using MODIS MOD13Q1 (16-day composites, 250m).
        
        Args:
            date (datetime): The date to fetch NDVI for. Defaults to now.
            
        Returns:
            float or None: NDVI value (scaled from -1 to 1), or None if unavailable.
        """
        if date is None:
            date = datetime(2025, 1, 1)

        start_date = ee.Date(date.strftime('%Y-%m-%d'))
        end_date = start_date.advance(16, 'day')  # MODIS product cycle

        ndvi_collection = ee.ImageCollection("MODIS/061/MOD13Q1") \
            .filterDate(start_date, end_date) \
            .filterBounds(self.point)

        ndvi_image = ndvi_collection.first()

        if ndvi_image is None:
            return None

        try:
            ndvi_sample = ndvi_image.select("NDVI").sample(self.point, 250).first()
            ndvi_value = ndvi_sample.get("NDVI").getInfo()
            return ndvi_value / 10000  # MODIS NDVI is scaled by 10,000
        except Exception as e:
            print("NDVI error:", e)
            return None

    def get_soil_properties(self):
        """
        Retrieves surface soil properties from SoilGrids (0–5 cm depth).
        
        Returns:
            dict: pH, organic carbon, bulk density, clay, sand, and silt %
        """
        depth = '0-5cm'
        props = {
            "phh2o": "pH_H2O",
            "ocd": "OrganicCarbon_g_per_kg",
            "bdod": "BulkDensity_kg_per_m3",
            "clay": "Clay_percent",
            "sand": "Sand_percent",
            "silt": "Silt_percent"
        }

        result = {}

        for key, label in props.items():
            try:
                image = ee.Image(f"projects/soilgrids-isric/{key}_mean")
                band_name = f"{key}_{depth}_mean"
                value = image.select(band_name).sample(self.point, 250).first().get(band_name).getInfo()
                result[label] = value
            except Exception as e:
                result[label] = None
                print(f"Soil property {label} not available:", e)

        return result

    def get_all_data(self, date=None):
        """
        Retrieve all available data for the location, including elevation, slope,
        aspect, and Dynamic World land cover (for the specified date).
        
        Args:
            date (datetime, optional): The date to use for land cover data.
                                       If None, the current UTC date is used.
        
        Returns:
            dict: A dictionary containing 'elevation', 'slope', 'aspect', and 'land_cover'.
        """
        terrain_data = self.get_elevation_and_terrain()
        land_cover = self.get_land_cover_dynamic_world(date=date)
        ndvi = self.get_ndvi_modis(date=date)
        ndvi_sentinel2 = self.get_ndvi_sentinel2(date=date)
        soil_properties = self.get_soil_properties()
        data = {**terrain_data, "land_cover": land_cover, "soil_properties": soil_properties, "ndvi": ndvi, "ndvi_sentinel2": ndvi_sentinel2}
        if land_cover is not None:
            data["land_cover_label"] = self.decode_land_cover_label(land_cover)

        return data
    

  

    @staticmethod
    def safe_sample(image, band_name, geom, scale):
        sample = image.sample(geom, scale).first()
        return ee.Algorithms.If(sample, sample.get(band_name), None)

    @classmethod
    def batch_process_locations(cls, points, date=datetime(2025, 1, 1)):
        ee.Initialize(project="ee-amizorach")

        features = [ee.Feature(ee.Geometry.Point(lon, lat), {'id': i}) for i, (lat, lon) in enumerate(points)]
        fc = ee.FeatureCollection(features)

        srtm = ee.Image("USGS/SRTMGL1_003")
        terrain = ee.Terrain.products(srtm)
        dynamic_world = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1") \
            .filterDate(date.strftime('%Y-%m-%d'), ee.Date(date).advance(1, 'month')) \
            .mosaic()

        ndvi_image = ee.ImageCollection("MODIS/061/MOD13Q1") \
            .filterDate(date.strftime('%Y-%m-%d'), ee.Date(date).advance(16, 'day')) \
            .first() \
            .select("NDVI") \
            .multiply(0.0001)

        def enrich_feature(f):
            geom = f.geometry()
            elev = cls.safe_sample(srtm, 'elevation', geom, 30)
            slope = cls.safe_sample(terrain, 'slope', geom, 30)
            aspect = cls.safe_sample(terrain, 'aspect', geom, 30)
            land_cover = cls.safe_sample(dynamic_world.select("label"), "label", geom, 10)
            ndvi = cls.safe_sample(ndvi_image, "NDVI", geom, 250)

            return f.set({
                'elevation': elev,
                'slope': slope,
                'aspect': aspect,
                'land_cover': land_cover,
                'ndvi': ndvi,
            })

        def add_soil_to_feature(f):
            geom = f.geometry()
            scale = 250

            def sample_soil(key):
                band = f"{key}_0-5cm_mean"
                img = ee.Image(f"projects/soilgrids-isric/{key}_mean").select(band)
                return cls.safe_sample(img, band, geom, scale)

            return f.set({
                'soil_ph': sample_soil('phh2o'),
                'soil_ocd': sample_soil('ocd'),
                'soil_bdod': sample_soil('bdod'),
                'soil_clay': sample_soil('clay'),
                'soil_sand': sample_soil('sand'),
                'soil_silt': sample_soil('silt')
            })

        enriched_fc = fc.map(enrich_feature).map(add_soil_to_feature)
        enriched_list = enriched_fc.getInfo()

        results = []
        for f in enriched_list['features']:
            props = f['properties']
            geom = f['geometry']['coordinates']
            results.append({
                'lat': geom[1],
                'lon': geom[0],
                'elevation': props.get('elevation'),
                'slope': props.get('slope'),
                'aspect': props.get('aspect'),
                'land_cover': props.get('land_cover'),
                'land_cover_label': cls.decode_land_cover_label(props.get('land_cover')),
                'ndvi': props.get('ndvi'),
                'soil': {
                    'soil_ph': props.get('soil_ph'),
                    'soil_ocd': props.get('soil_ocd'),
                    'soil_bdod': props.get('soil_bdod'),
                    'soil_clay': props.get('soil_clay'),
                    'soil_sand': props.get('soil_sand'),
                    'soil_silt': props.get('soil_silt')
                }
            })

        return results
    @classmethod
    def add_soil_to_feature(cls, f):
        geom = f.geometry()
        scale = 250  # sampling resolution

        def sample(band_img, band_name):
            return band_img.sample(geom, scale).first().get(band_name)

        try:
            pH = sample(ee.Image("projects/soilgrids-isric/phh2o_mean"), "phh2o_0-5cm_mean")
            OC = sample(ee.Image("projects/soilgrids-isric/ocd_mean"), "ocd_0-5cm_mean")
            BD = sample(ee.Image("projects/soilgrids-isric/bdod_mean"), "bdod_0-5cm_mean")
            clay = sample(ee.Image("projects/soilgrids-isric/clay_mean"), "clay_0-5cm_mean")
            sand = sample(ee.Image("projects/soilgrids-isric/sand_mean"), "sand_0-5cm_mean")
            silt = sample(ee.Image("projects/soilgrids-isric/silt_mean"), "silt_0-5cm_mean")
        except Exception:
            return f

        return f.set({
            'soil_ph': pH,
            'soil_ocd': OC,
            'soil_bdod': BD,
            'soil_clay': clay,
            'soil_sand': sand,
            'soil_silt': silt
        })



# Example usage:
if __name__ == "__main__":
    # Example: San Francisco, CA.
    lat = 36.73582
    lon = -119.04663
    lat2 = 36.73582
    lon2 = -119.04663
    lat3 = 36.73582
    lon3 = -119.04663
    points = [(lat, lon), (lat2, lon2), (lat3, lon3)]
    data = GELocationUtils.batch_process_locations(points)
    print("GGELocation Data:", data)
    # ge = GEELocationUtils(lat, lon)
    # print("GGELocation Data:", ge.get_all_data().keys())
