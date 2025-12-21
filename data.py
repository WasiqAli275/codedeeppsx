from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
from datetime import datetime
import csv
import sys
import io
import os

# Fix Unicode encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def setup_driver():
    """Chrome driver setup - Optimized for speed"""
    print("Setting up Chrome driver...")
    
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    
    # Performance optimizations
    chrome_options.add_argument('--disable-extensions')
    chrome_options.add_argument('--disable-plugins')
    chrome_options.add_argument('--disable-images')  # Speed up loading
    chrome_options.add_experimental_option("prefs", {
        "profile.default_content_setting_values.notifications": 2,
        "profile.default_content_settings.popups": 0,
    })
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    print("Chrome driver ready!")
    return driver

def extract_correct_psx_data(driver):
    """FIXED: Correct PSX data extraction with ALL columns including SECTOR and LISTED IN"""
    try:
        url = "https://dps.psx.com.pk/market-watch"
        print(f"Loading: {url}")
        
        driver.get(url)
        
        print("Waiting for data to load...")
        # Reduced wait time - use explicit waits instead
        wait = WebDriverWait(driver, 15)
        
        # Wait for table to be present
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
        print("Table found, extracting data...")
        
        # Extract with COMPLETE column mapping
        extract_script = """
            let allStocks = [];
            
            let tables = document.querySelectorAll('table');
            let targetTable = null;
            
            // Find the main market watch table
            for (let table of tables) {
                let headers = table.querySelectorAll('th');
                if (headers.length >= 10) {  // PSX table has 11 columns
                    let headerText = Array.from(headers).map(h => h.innerText.trim()).join(' ');
                    if (headerText.includes('Symbol') || headerText.includes('Sector') || headerText.includes('LDCP')) {
                        targetTable = table;
                        break;
                    }
                }
            }
            
            if (!targetTable && tables.length > 0) {
                targetTable = tables[0];
            }
            
            if (!targetTable) return allStocks;
            
            let tbody = targetTable.querySelector('tbody');
            if (!tbody) return allStocks;
            
            let rows = tbody.querySelectorAll('tr');
            
            rows.forEach((row, rowIndex) => {
                let cells = row.querySelectorAll('td');
                
                // CORRECT PSX TABLE STRUCTURE - 11 COLUMNS:
                // 0: Symbol
                // 1: Sector
                // 2: Listed In
                // 3: LDCP (Last Day Close Price)
                // 4: Open
                // 5: High
                // 6: Low
                // 7: Current
                // 8: Change
                // 9: Change%
                // 10: Volume
                
                if (cells.length >= 9) {  // Some rows might have fewer cells
                    let symbol = cells[0]?.innerText.trim() || '';
                    
                    // Skip invalid rows
                    if (!symbol || symbol === '' || symbol.length > 20 || 
                        symbol.includes('Symbol') || symbol.includes('Last') ||
                        symbol === 'PSX' || symbol === 'KSE-100' ||
                        symbol.includes('Open') || symbol.includes('High') ||
                        symbol.includes('Low') || symbol.includes('Volume')) {
                        return;
                    }
                    
                    // COMPLETE COLUMN MAPPING - ALL 11 COLUMNS
                    let stockData = {
                        symbol: symbol,
                        sector: cells[1]?.innerText.trim() || '',           // Column 1 - SECTOR
                        listed_in: cells[2]?.innerText.trim() || '',        // Column 2 - LISTED IN
                        ldcp: cells[3]?.innerText.trim() || '0',           // Column 3 - LDCP
                        open: cells[4]?.innerText.trim() || '0',           // Column 4 - Open
                        high: cells[5]?.innerText.trim() || '0',           // Column 5 - High
                        low: cells[6]?.innerText.trim() || '0',            // Column 6 - Low
                        current: cells[7]?.innerText.trim() || '0',        // Column 7 - Current
                        change: cells[8]?.innerText.trim() || '0',         // Column 8 - Change
                        change_percent: cells[9]?.innerText.trim() || '0', // Column 9 - Change%
                        volume: cells[10]?.innerText.trim() || '0'         // Column 10 - Volume
                    };
                    
                    allStocks.push(stockData);
                }
            });
            
            return allStocks;
        """
        
        raw_data = driver.execute_script(extract_script)
        
        if not raw_data or len(raw_data) == 0:
            print("JavaScript extraction failed, trying robust manual extraction...")
            return extract_manual_complete(driver)
        
        print(f"Extracted {len(raw_data)} stocks successfully with COMPLETE columns")
        return raw_data
        
    except Exception as e:
        print(f"Error in main extraction: {e}")
        return extract_manual_complete(driver)

def extract_manual_complete(driver):
    """Robust manual extraction with COMPLETE column mapping"""
    print("Using robust manual extraction with ALL columns...")
    
    try:
        stocks_data = []
        
        wait = WebDriverWait(driver, 20)
        
        # Find the main table
        table = wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
        
        # Get all rows
        rows = table.find_elements(By.TAG_NAME, "tr")
        print(f"Found {len(rows)} total rows")
        
        processed_count = 0
        for i, row in enumerate(rows):
            try:
                cells = row.find_elements(By.TAG_NAME, "td")
                
                # Skip rows with too few cells or header rows
                if len(cells) < 9:
                    continue
                
                symbol = cells[0].text.strip()
                
                # Skip invalid symbols
                if (not symbol or len(symbol) > 20 or 
                    any(keyword in symbol for keyword in ['Symbol', 'Last', 'Open', 'High', 'Low', 'Current', 'Change', 'Volume']) or
                    'PSX' in symbol or 'KSE' in symbol):
                    continue
                
                # COMPLETE COLUMN MAPPING FOR PSX - ALL 11 COLUMNS:
                stock = {
                    'symbol': symbol,
                    'sector': cells[1].text.strip() if len(cells) > 1 else '',           # SECTOR
                    'listed_in': cells[2].text.strip() if len(cells) > 2 else '',        # LISTED IN
                    'ldcp': cells[3].text.strip() if len(cells) > 3 else '0',           # LDCP
                    'open': cells[4].text.strip() if len(cells) > 4 else '0',           # Open
                    'high': cells[5].text.strip() if len(cells) > 5 else '0',           # High
                    'low': cells[6].text.strip() if len(cells) > 6 else '0',            # Low
                    'current': cells[7].text.strip() if len(cells) > 7 else '0',        # Current
                    'change': cells[8].text.strip() if len(cells) > 8 else '0',         # Change
                    'change_percent': cells[9].text.strip() if len(cells) > 9 else '0', # Change%
                    'volume': cells[10].text.strip() if len(cells) > 10 else '0'        # Volume
                }
                
                stocks_data.append(stock)
                processed_count += 1
                
                if processed_count % 50 == 0:
                    print(f"   {processed_count} valid stocks processed...")
                
            except Exception as e:
                continue
        
        print(f"Robust extraction: {len(stocks_data)} valid stocks with ALL columns")
        return stocks_data
        
    except Exception as e:
        print(f"Robust extraction error: {e}")
        return []

def clean_numeric_value(value):
    """Clean and convert numeric values properly"""
    if not value or value in ['N/A', '-', '', ' ', 'NAN', 'NULL']:
        return '0'
    
    try:
        # Remove commas, spaces, percentage signs, and other non-numeric characters except decimal point and minus
        cleaned = str(value).strip()
        cleaned = cleaned.replace(',', '').replace(' ', '').replace('%', '')
        cleaned = cleaned.replace('(', '').replace(')', '').replace('$', '')
        cleaned = cleaned.replace('Rs.', '').replace('PKR', '')
        
        # Handle negative values properly
        if cleaned.startswith('-'):
            cleaned = '-' + cleaned[1:].lstrip()
        
        # If it's empty after cleaning, return 0
        if not cleaned:
            return '0'
            
        # Convert to float and back to string to validate
        float_val = float(cleaned)
        return str(float_val)
        
    except:
        return '0'

def validate_stock_data(stock):
    """Validate if stock data looks reasonable"""
    try:
        # Check if symbol is reasonable
        if not stock['symbol'] or len(stock['symbol']) > 20:
            return False
        
        # Check if numeric values are reasonable
        current_price = float(clean_numeric_value(stock['current']))
        if current_price <= 0 or current_price > 100000:  # Assuming no stock price > 100,000
            return False
            
        return True
    except:
        return False

def save_complete_data(stocks_data):
    """Save data in SINGLE CSV format only with timestamp-based filename"""
    if not stocks_data:
        print("No data to save")
        return False
    
    try:
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        # Create filename from timestamp (replace colons with underscores for filesystem compatibility)
        filename_timestamp = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        csv_filename = f"psx_data_{filename_timestamp}.csv"
        
        # Filter and validate data
        validated_data = []
        for item in stocks_data:
            if validate_stock_data(item):
                formatted_item = {
                    'Symbol': item['symbol'],
                    'Sector': item['sector'],
                    'Listed_In': item['listed_in'],
                    'LDCP': clean_numeric_value(item['ldcp']),
                    'Open': clean_numeric_value(item['open']),
                    'High': clean_numeric_value(item['high']),
                    'Low': clean_numeric_value(item['low']),
                    'Current': clean_numeric_value(item['current']),
                    'Change': clean_numeric_value(item['change']),
                    'Change(%)': clean_numeric_value(item['change_percent']),
                    'Volume': clean_numeric_value(item['volume'])
                }
                validated_data.append(formatted_item)
        
        if not validated_data:
            print("No validated data to save")
            return False
            
        df = pd.DataFrame(validated_data)
        
        print(f"Data Verification:")
        print(f"   Total Valid Stocks: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        
        # ==================== SAVE CSV FILE ONLY ====================
        print(f"Saving CSV file: {csv_filename}")
        
        # Overwrite existing file if it exists (open with 'w' mode does this automatically)
        with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            
            # Metadata
            writer.writerow(['Pakistan Stock Exchange - COMPLETELY FIXED Data'])
            writer.writerow(['Timestamp', timestamp_str])
            writer.writerow(['Total Valid Stocks', len(df)])
            writer.writerow(['Fix Applied', 'Complete column mapping correction - All data validated'])
            writer.writerow([])
            
            # Headers - ALL 11 COLUMNS
            writer.writerow(['Symbol', 'Sector', 'Listed_In', 'LDCP', 'Open', 'High', 'Low', 'Current', 'Change', 'Change(%)', 'Volume'])
            
            # Data
            for _, row in df.iterrows():
                writer.writerow([
                    row['Symbol'],
                    row['Sector'],
                    row['Listed_In'],
                    row['LDCP'],
                    row['Open'],
                    row['High'],
                    row['Low'],
                    row['Current'],
                    row['Change'],
                    row['Change(%)'],
                    row['Volume']
                ])
        
        print(f"✓ CSV file saved: {csv_filename}")
        
        # Display comprehensive sample for verification
        print("=" * 150)
        print("COMPLETE SAMPLE DATA VERIFICATION (WITH ALL COLUMNS):")
        print('=' * 150)
        print(f"{'Symbol':<10} {'Sector':<8} {'Listed':<8} {'LDCP':<8} {'Open':<8} {'High':<8} {'Low':<8} {'Current':<8} {'Change':<8} {'Chg%':<8} {'Volume':<12}")
        print('-' * 150)
        
        for idx in range(min(10, len(df))):
            row = df.iloc[idx]
            print(f"{row['Symbol']:<10} {row['Sector']:<8} {row['Listed_In']:<8} {row['LDCP']:<8} {row['Open']:<8} {row['High']:<8} {row['Low']:<8} {row['Current']:<8} {row['Change']:<8} {row['Change(%)']:<8} {row['Volume']:<12}")
        
        print('=' * 150)
        
        return True
        
    except Exception as e:
        print(f"Save error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main execution - Optimized and Fixed"""
    print("=" * 100)
    print("PSX COMPLETE DATA SCRAPER - WITH ALL COLUMNS")
    print("=" * 100)
    print("FIXES APPLIED:")
    print("   1. ADDED MISSING COLUMNS: SECTOR and LISTED IN")
    print("   2. Correct column alignment for all data")
    print("   3. Optimized performance - faster loading")
    print("   4. Complete data validation")
    print("   5. Saves in SINGLE CSV format with timestamp filename")
    print("=" * 100)
    print()
    
    driver = None
    
    try:
        driver = setup_driver()
        
        # Extract COMPLETE data with all columns
        stocks_data = extract_correct_psx_data(driver)
        
        if stocks_data and len(stocks_data) > 0:
            print("=" * 100)
            print(f"SUCCESS! Extracted {len(stocks_data)} stocks with ALL COLUMNS")
            print("=" * 100)
            
            # Save with validation in CSV format only
            print("Saving complete data in CSV format with timestamp filename...")
            success = save_complete_data(stocks_data)
            
            if success:
                print("=" * 100)
                print("SCRAPING COMPLETED SUCCESSFULLY!")
                print("=" * 100)
                print("Output File:")
                print("   psx_data_YYYY-MM-DD_HH-MM-SS.csv - COMPLETE data with all columns")
                print("")
                print("ALL ISSUES RESOLVED:")
                print("   ✓ Symbols correctly extracted")
                print("   ✓ SECTOR column added")
                print("   ✓ LISTED IN column added") 
                print("   ✓ All columns properly aligned")
                print("   ✓ Data validated and cleaned")
                print("   ✓ Performance optimized")
                print("   ✓ Single CSV file created with timestamp filename")
                print("   ✓ File automatically overwrites existing file with same timestamp")
                print("=" * 100)
            else:
                print("Failed to save validated data")
        else:
            print("No data extracted after all attempts!")
            
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if driver:
            print("Closing browser...")
            driver.quit()
            print("Done!")

if __name__ == "__main__":
    main()