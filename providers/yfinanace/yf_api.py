import yfinance as yf
import matplotlib.pyplot as plt
# Define the ticker and parameters
ticker = "GOOG"  # Example: Apple Inc.
data = yf.download(
    tickers=ticker,
    period="20d",       # Last 30 days
    interval="1h",      # Hourly data
    auto_adjust=True    # Adjust for splits/dividends if needed
)

# Display the first few rows
print(data.head())
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Close Price')
plt.title(f"{ticker} - Hourly Close Price (Last 30 Days)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
input("Press Enter to exit...")

