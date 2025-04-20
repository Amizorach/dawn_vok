from polygon import RESTClient
from scapy.all import *
class PolygonAPI:
    def __init__(self, api_key):
        self.client = RESTClient(api_key)

if __name__ == '__main__':

# Load packets from a pcap file
    packets = rdpcap('raw_data/t1.pcap')

    # Print basic info for the first 5 packets
    for pkt in packets[:5]:
        print(pkt.summary())

#     polygon_api = PolygonAPI(api_key='LEuAR3YTDFFYjT3TFexUOmHi6hjfo3BQ')
    
#  #   print(polygon_api.client.get_aggs('AAPL', multiplier=1, timespan='day', from_='2024-01-01', to='2024-01-11', adjusted=True))
#     aggs = []
#     for a in polygon_api.client.list_aggs(
#         "AAPL",
#         1,
#         "minute",
#         "2024-01-09",
#         "2024-01-11",
#         adjusted="true",
#         sort="asc",
#         limit=120,
#     ):
#         aggs.append(a)

#     print(aggs) # tickers = []
#     # for t in polygon_api.client.list_tickers(
#     #     market="stocks",
#     #     active="true",
#     #     order="asc",
#     #     limit="100",
#     #     sort="ticker",
#     #     ):
#     #     tickers.append(t)

#     # print(tickers)
