    
import cdsapi


dataset = "sis-agrometeorological-indicators"
request = {
    "variable": "2m_temperature",
    "statistic": [
      
        "day_time_maximum",
        "day_time_mean",
       
    ],
    "year": ["2024",],
    "month": [
        "01", 
        
    ],
    "day": [
        "01", 
    ],
    "version": "1_1",
    "area": [32.5, 35.4, 31, 35.6]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
