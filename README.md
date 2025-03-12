# MLB Pitcher Strikeout Prediction Model

This project implements a machine learning model to predict strikeouts (SO) for MLB (Major League Baseball) pitchers, using historical data and betting odds.

## Project Structure

```
pitcher_so_ML/
├── web_scrapping/
│   ├── request_odds_V1.py
│   └── request_SO_odds.py
└── src/
    ├── Batting Teams.ipynb
    ├── Pitcher ML V1 WebScrapping.ipynb
    └── ML Strikeout.ipynb
```

## Description

The project consists of three main components:

1. **Web Scraping (web_scrapping/)**
   - `request_SO_odds.py`: Script to collect strikeout odds from BettingPros API
   - `request_odds_V1.py`: Initial version of the odds collection script

2. **Analysis and Modeling (src/)**
   - `ML Strikeout.ipynb`: Main notebook with machine learning model implementation
   - `Batting Teams.ipynb`: Analysis of batting teams data
   - `Pitcher ML V1 WebScrapping.ipynb`: Notebook with initial analysis and web scraping

## Features

### Data Collection
- Automatic collection of strikeout odds from BettingPros API
- Processing of historical pitcher data
- Integration with team strikeout statistics

### Machine Learning Model
- Uses Random Forest Regressor for predictions
- Features include:
  - IP (Innings Pitched)
  - H (Hits)
  - BB (Base on Balls)
  - ERA (Earned Run Average)
  - FIP (Fielding Independent Pitching)
  - %K (Opponent Team's Strikeout Percentage)

### Model Characteristics
- Performance weighting based on:
  - Current season (40%)
  - Last 5 games (30%)
  - Previous season (30%)
- Cross-validation for hyperparameter optimization
- Prediction confidence calculation

## Requirements

- Python 3.x
- Main libraries:
  - pandas
  - numpy
  - scikit-learn
  - requests

## How to Use

1. **Data Collection**
```python
python web_scrapping/request_SO_odds.py
```

2. **Model Training**
- Open the `ML Strikeout.ipynb` notebook
- Execute the cells to train the model

3. **Making Predictions**
- Use the `predict_strikeouts_with_confidence()` function for new predictions
- Provide:
  - Pitcher name
  - Opponent team
  - Strikeout line

## Results

The model provides:
- Strikeout prediction
- Recommendation (Over/Under)
- Prediction confidence percentage

## Notes

- Model is updated daily with new data
- Takes into account historical statistics and recent trends
- Includes adjustments for different stadiums and opponent teams 