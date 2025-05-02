import subprocess
import os
import sys
from datetime import date, timedelta
import logging

# --- Configuration ---
START_YEAR = 2021
START_MONTH = 1
# Define the end date explicitly
END_YEAR = 2025
END_MONTH = 3 

OUTPUT_DIR = "output" # Should match the default in cli.py or be configurable
PYTHON_EXECUTABLE = sys.executable # Use the same python that runs this script
CLI_SCRIPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cli.py')) # Path to cli.py relative to this script

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("batch_export.log"),
        logging.StreamHandler(sys.stdout) # Also print logs to console
    ]
)
logger = logging.getLogger(__name__)

def get_months_range(start_year, start_month, end_year, end_month):
    """Generates year-month tuples from start date up to and including the end month."""
    # current_date = date.today() # No longer needed
    # current_year = current_date.year
    # current_month = current_date.month
    
    year = start_year
    month = start_month
    
    # Loop while the generated year/month is less than or equal to the end year/month
    while (year < end_year) or (year == end_year and month <= end_month):
        yield year, month
        month += 1
        if month > 12:
            month = 1
            year += 1

def run_export_for_month(year, month):
    """Runs the cli.py export command for a specific month."""
    month_str = f"{year}-{month:02d}"
    command = [
        PYTHON_EXECUTABLE, 
        CLI_SCRIPT_PATH, 
        "--export", 
        "--export-month", 
        month_str,
        "--output-dir", # Explicitly specify output dir for clarity
        OUTPUT_DIR 
        # Add --verbose if needed: , "--verbose"
    ]
    
    logger.info(f"Running export for month: {month_str}")
    logger.debug(f"Executing command: {' '.join(command)}")
    
    try:
        # Ensure output directory exists before cli.py tries to use it
        os.makedirs(OUTPUT_DIR, exist_ok=True) 
        
        result = subprocess.run(
            command, 
            check=True, # Raises CalledProcessError if command returns non-zero exit code
            capture_output=True, # Capture stdout/stderr
            text=True # Decode stdout/stderr as text
        )
        logger.info(f"Successfully exported {month_str}.")
        logger.debug(f"cli.py stdout:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Export failed for month {month_str}. Return code: {e.returncode}")
        logger.error(f"cli.py stderr:\n{e.stderr}")
        logger.error(f"cli.py stdout:\n{e.stdout}") # Also log stdout on error
        return False
    except FileNotFoundError:
        logger.error(f"Error: Could not find Python executable '{PYTHON_EXECUTABLE}' or script '{CLI_SCRIPT_PATH}'.")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during export for {month_str}: {e}", exc_info=True)
        return False

def main():
    logger.info("--- Starting Batch Export Process ---")
    logger.info(f"Exporting months from {START_YEAR}-{START_MONTH:02d} up to {END_YEAR}-{END_MONTH:02d}.")
    successful_exports = 0
    failed_exports = 0
    
    # Pass end year/month to the generator
    for year, month in get_months_range(START_YEAR, START_MONTH, END_YEAR, END_MONTH):
        if run_export_for_month(year, month):
            successful_exports += 1
        else:
            failed_exports += 1
            # Decide whether to continue or stop on failure
            # For now, let's continue to export other months
            logger.warning(f"Continuing export process after failure for {year}-{month:02d}.")

    logger.info("--- Batch Export Process Finished ---")
    logger.info(f"Successful exports: {successful_exports}")
    logger.info(f"Failed exports: {failed_exports}")
    
    if failed_exports > 0:
        logger.warning("Some monthly exports failed. Check logs above for details.")
        sys.exit(1) # Exit with error code if any exports failed
    else:
        logger.info("All monthly exports completed successfully.")
        sys.exit(0)

if __name__ == "__main__":
    main() 