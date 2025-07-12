#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fully_pitchers_record import run_pitcher_scraper

if __name__ == "__main__":
    print("Testing improved pitcher scraper...")
    run_pitcher_scraper() 