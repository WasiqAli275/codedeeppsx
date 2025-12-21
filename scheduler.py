# scheduler.py - Cloud Scheduler
import schedule
import time
from datetime import datetime
from scraper_cloud import CloudScraper
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper_scheduler.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def scheduled_scrape():
    """Function to run on schedule"""
    logger.info("Running scheduled scrape")
    try:
        scraper = CloudScraper()
        success = scraper.run_scrape()
        
        if success:
            logger.info("Scheduled scrape completed successfully")
        else:
            logger.error("Scheduled scrape failed")
            
    except Exception as e:
        logger.error(f"Error in scheduled scrape: {str(e)}")

def main():
    """Run the scheduler"""
    logger.info("Starting PSX Cloud Scraper Scheduler")
    
    # Schedule every 5 minutes
    schedule.every(5).minutes.do(scheduled_scrape)
    
    # Also run immediately on start
    scheduled_scrape()
    
    logger.info("Scheduler started. Running every 5 minutes...")
    
    # Keep running
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
            break
        except Exception as e:
            logger.error(f"Scheduler error: {str(e)}")
            time.sleep(60)

if __name__ == "__main__":
    main()