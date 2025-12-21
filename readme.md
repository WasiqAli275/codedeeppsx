# supabass 

you need to enter the code on the supabass


``` sql
-- Run this in your Supabase SQL editor
CREATE TABLE IF NOT EXISTS stock_data (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    sector TEXT,
    listed_in TEXT,
    ldcp FLOAT,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    current FLOAT,
    change FLOAT,
    change_percent FLOAT,
    volume BIGINT,
    scrape_time TIMESTAMP WITH TIME ZONE NOT NULL,
    dataset_name TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()),
    
    -- Ensure no duplicate symbol entries for the same scrape time
    UNIQUE(symbol, scrape_time)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_stock_data_scrape_time ON stock_data(scrape_time DESC);
CREATE INDEX IF NOT EXISTS idx_stock_data_symbol ON stock_data(symbol);
CREATE INDEX IF NOT EXISTS idx_stock_data_symbol_scrape_time ON stock_data(symbol, scrape_time DESC);

-- Enable Row Level Security (optional)
ALTER TABLE stock_data ENABLE ROW LEVEL SECURITY;

-- Create policy for public read access (adjust as needed)
CREATE POLICY "Allow public read access" ON stock_data
    FOR SELECT USING (true);
```


Option A: Deploy to Streamlit Cloud:
Push code to GitHub repository

Go to share.streamlit.io

Connect your GitHub repository

Add secrets in Streamlit Cloud:

SUPABASE_URL = Your Supabase project URL

SUPABASE_KEY = Your Supabase anon/public key

Option B: Deploy to Google Cloud Run with Scheduler:
Create Dockerfile:

dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
Deploy to Cloud Run:

bash
gcloud run deploy psx-stock-analyzer \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
Setup Cloud Scheduler for scraping:

bash
### Create Cloud Function for scraping

gcloud functions deploy psx-scraper \
  --runtime python39 \
  --trigger-http \
  --allow-unauthenticated

### Schedule every 5 minutes

gcloud scheduler jobs create http psx-scrape-job \
  --schedule="*/5 * * * *" \
  --uri="YOUR_CLOUD_FUNCTION_URL" \
  --http-method=POST
Key Features Implemented:
☁️ Cloud Architecture:

Supabase for data storage

Auto-scraping every 5 minutes

No local script execution

⏰ Time Interval Selection:

5m, 15m, 1h, 4h, 1D intervals

Timestamp selection for each interval

Smart aggregation for higher timeframes

📊 Volume Delta Calculations:

Δ Volume vs previous interval

Percentage change calculation

Color-coded positive/negative deltas

🎯 Advanced Filtering:

Filter by gainers/losers

High/low volume filtering

Multiple sorting options

📈 Enhanced Visualizations:

Volume delta charts

Performance scatter plots

Interactive Plotly graphs

⚡ Real-time Updates:

Data freshness indicator

Auto-refresh capability

Cloud-based updates

Testing the Migration:
Test Supabase Connection:

python
### test_supabase.py

```python
from supabase import create_client
import os

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

client = create_client(url, key)
print("Connection successful!")
```

Test Data Flow:

Run python scheduler.py locally

Check data appears in Supabase

Launch Streamlit app to view data

This implementation provides a complete cloud migration with all requested features including time-based aggregation, volume delta calculations, and robust error handling.

install the required library 

pip
``` python
pip install supabase python-dotenv schedule
```

and also add the required information in the env file 
``` python

SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_key_here
```
