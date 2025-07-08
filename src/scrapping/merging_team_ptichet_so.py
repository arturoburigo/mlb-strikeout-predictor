#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to merge pitchers_data.csv and team_strikeout_percentage.csv tables
Adds the opp_so_avg column based on the opponent team's strikeout percentage
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

def merge_pitcher_team_strikeout_data():
    """
    Merges pitcher data tables with team strikeout percentage data
    """
    
    print("🔄 Starting table merge...")
    
    # Loading data
    print("📊 Loading pitcher data...")
    pitchers_df = pd.read_csv('../../pitchers_detailed_data.csv')
    
    print("📊 Loading team strikeout percentage data...")
    team_so_df = pd.read_csv('../../team_strikeout_percentage.csv')
    
    # Data cleaning
    print("🧹 Cleaning data...")
    
    # Removing asterisks from Pitcher_Name column
    if 'Pitcher_Name' in pitchers_df.columns:
        pitchers_df['Pitcher_Name'] = pitchers_df['Pitcher_Name'].str.replace('*', '', regex=False)
        print("✅ Asterisks removed from Pitcher_Name column")
    
    # Cleaning date format (removing numbers in parentheses)
    date_columns = [col for col in pitchers_df.columns if 'date' in col.lower() or 'Date' in col]
    for col in date_columns:
        if col in pitchers_df.columns:
            # Remove patterns like "2023-07-01 (1)" or "2023-07-01 (2)"
            pitchers_df[col] = pitchers_df[col].astype(str).str.replace(r'\s*\(\d+\)', '', regex=True)
            print(f"✅ Date format cleaned in column {col}")
    
    # Checking for other date columns that might have the same issue
    # Looking for columns containing dates in yyyy-mm-dd format
    for col in pitchers_df.columns:
        if pitchers_df[col].dtype == 'object':
            # Check if the column contains strings that look like dates
            sample_values = pitchers_df[col].dropna().astype(str).head(10)
            if any(re.match(r'\d{4}-\d{2}-\d{2}', str(val)) for val in sample_values):
                pitchers_df[col] = pitchers_df[col].astype(str).str.replace(r'\s*\(\d+\)', '', regex=True)
                print(f"✅ Date format cleaned in column {col}")
    
    # Displaying cleaning examples
    if 'Pitcher_Name' in pitchers_df.columns:
        print(f"\n📊 Examples of pitcher names after cleaning:")
        sample_names = pitchers_df['Pitcher_Name'].dropna().head(5)
        for name in sample_names:
            print(f"  - {name}")
    
    # Checking for problematic date formats
    date_columns = [col for col in pitchers_df.columns if 'date' in col.lower() or 'Date' in col]
    for col in date_columns:
        if col in pitchers_df.columns:
            sample_dates = pitchers_df[col].dropna().astype(str).head(5)
            print(f"\n📊 Date examples in column {col}:")
            for date in sample_dates:
                print(f"  - {date}")
    
    # Displaying basic information
    print(f"\n📋 Data information:")
    print(f"Pitchers data: {pitchers_df.shape[0]} rows, {pitchers_df.shape[1]} columns")
    print(f"Team strikeout data: {team_so_df.shape[0]} rows, {team_so_df.shape[1]} columns")
    
    print(f"\n🔍 Pitcher data columns:")
    print(pitchers_df.columns.tolist())
    
    print(f"\n🔍 Team strikeout data columns:")
    print(team_so_df.columns.tolist())
    
    # Checking unique values in join columns
    print(f"\n📊 Unique values in 'Season' (pitchers): {sorted(pitchers_df['Season'].unique())}")
    print(f"📊 Unique values in 'Opp' (pitchers): {sorted(pitchers_df['Opp'].unique())}")
    print(f"📊 Unique values in 'Team' (strikeout): {sorted(team_so_df['Team'].unique())}")
    
    # Preparing strikeout dataframe for merge
    print("\n🔄 Preparing strikeout data for merge...")
    
    # Transforming strikeout dataframe from wide to long format
    team_so_long = team_so_df.melt(
        id_vars=['Team'], 
        value_vars=['2023', '2024', '2025'], 
        var_name='Season', 
        value_name='opp_so_avg'
    )
    
    # Formatting opp_so_avg column to two decimal places
    team_so_long['opp_so_avg'] = team_so_long['opp_so_avg'].round(2)
    
    # Converting Season to string to ensure compatibility
    team_so_long['Season'] = team_so_long['Season'].astype(str)
    pitchers_df['Season'] = pitchers_df['Season'].astype(str)
    
    print(f"📊 Team strikeout data after transformation: {team_so_long.shape[0]} rows")
    print(f"📊 Example of transformed data:")
    print(team_so_long.head(10))
    
    # Performing the merge
    print("\n🔗 Merging tables...")
    
    # Merge based on Season and Opp (opponent team)
    merged_df = pitchers_df.merge(
        team_so_long,
        left_on=['Season', 'Opp'],
        right_on=['Season', 'Team'],
        how='left'
    )
    
    # Removing duplicate Team column (if exists)
    if 'Team' in merged_df.columns:
        merged_df = merged_df.drop('Team', axis=1)
    
    # Checking merge results
    print(f"\n✅ Merge completed!")
    print(f"📊 Original data: {pitchers_df.shape[0]} rows")
    print(f"📊 Data after merge: {merged_df.shape[0]} rows")
    print(f"📊 Columns after merge: {merged_df.shape[1]} columns")
    
    # Checking null values in new column
    null_count = merged_df['opp_so_avg'].isnull().sum()
    print(f"📊 Null values in 'opp_so_avg': {null_count} ({null_count/len(merged_df)*100:.2f}%)")
    
    if null_count > 0:
        print("\n⚠️  Null values found. Checking possible causes...")
        
        # Checking which teams were not found
        null_opps = merged_df[merged_df['opp_so_avg'].isnull()]['Opp'].unique()
        print(f"📊 Teams not found in merge: {sorted(null_opps)}")
        
        # Checking for differences in team names
        pitchers_opps = set(pitchers_df['Opp'].unique())
        team_so_teams = set(team_so_df['Team'].unique())
        
        missing_in_team_so = pitchers_opps - team_so_teams
        missing_in_pitchers = team_so_teams - pitchers_opps
        
        if missing_in_team_so:
            print(f"📊 Teams in pitchers_data.csv that are not in team_strikeout_percentage.csv: {sorted(missing_in_team_so)}")
        
        if missing_in_pitchers:
            print(f"📊 Teams in team_strikeout_percentage.csv that are not in pitchers_data.csv: {sorted(missing_in_pitchers)}")
        
        # Handling team name mismatches with fallback mappings
        print("\n🔄 Attempting to fix team name mismatches...")
        
        # Define team name mappings for common mismatches
        team_mappings = {
            'ATH': 'OAK',  # Athletics
            'OAK': 'OAK',  # Oakland Athletics
        }
        
        # Create a copy of the original data for fallback merge
        pitchers_df_fallback = pitchers_df.copy()
        
        # Apply team name mappings to pitchers data
        for old_name, new_name in team_mappings.items():
            if old_name in pitchers_df_fallback['Opp'].values:
                pitchers_df_fallback.loc[pitchers_df_fallback['Opp'] == old_name, 'Opp'] = new_name
                print(f"✅ Mapped '{old_name}' to '{new_name}' in pitchers data")
        
        # Try merge again with mapped team names
        merged_df_fallback = pitchers_df_fallback.merge(
            team_so_long,
            left_on=['Season', 'Opp'],
            right_on=['Season', 'Team'],
            how='left'
        )
        
        # Remove duplicate Team column if exists
        if 'Team' in merged_df_fallback.columns:
            merged_df_fallback = merged_df_fallback.drop('Team', axis=1)
        
        # Check if fallback merge improved the results
        null_count_fallback = merged_df_fallback['opp_so_avg'].isnull().sum()
        improvement = null_count - null_count_fallback
        
        if improvement > 0:
            print(f"✅ Fallback merge successful! Reduced null values from {null_count} to {null_count_fallback} ({improvement} fixed)")
            merged_df = merged_df_fallback
            null_count = null_count_fallback
        else:
            print(f"⚠️  Fallback merge did not improve results. Null values remain: {null_count}")
    
    # Displaying statistics of new column
    print(f"\n📊 Statistics of 'opp_so_avg' column:")
    print(merged_df['opp_so_avg'].describe())
    
    # Displaying some example rows
    print(f"\n📊 Example of data after merge:")
    print(merged_df[['Season', 'Pitcher_Name', 'Opp', 'opp_so_avg']].head(10))
    
    # Saving results
    output_file = '../pitchers_data_with_opp_so.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"\n💾 Data saved to: {output_file}")
    
    # Checking merge quality
    print(f"\n🔍 Merge quality verification:")
    
    # Checking if all years were processed
    years_processed = merged_df['Season'].unique()
    print(f"📊 Processed years: {sorted(years_processed)}")
    
    # Checking distribution of new column by year
    print(f"\n📊 Distribution of 'opp_so_avg' by year:")
    for year in sorted(years_processed):
        year_data = merged_df[merged_df['Season'] == year]['opp_so_avg']
        print(f"  {year}: Mean = {year_data.mean():.2f}%, Median = {year_data.median():.2f}%")
    
    # Checking for extreme or suspicious values
    print(f"\n📊 Extreme values check:")
    print(f"  Minimum: {merged_df['opp_so_avg'].min():.2f}%")
    print(f"  Maximum: {merged_df['opp_so_avg'].max():.2f}%")
    
    # Checking if values make sense (between 50% and 80%)
    valid_range = merged_df[(merged_df['opp_so_avg'] >= 50) & (merged_df['opp_so_avg'] <= 80)]
    print(f"  Values in expected range (50-80%): {len(valid_range)}/{len(merged_df)} ({len(valid_range)/len(merged_df)*100:.2f}%)")
    
    return merged_df

def analyze_merge_results(df):
    """
    Analyzes the merge results
    """
    print(f"\n📈 MERGE RESULTS ANALYSIS")
    print("=" * 50)
    
    # Analysis by opponent team
    print(f"\n🏟️  Analysis by opponent team (Top 10 with most games):")
    opp_counts = df['Opp'].value_counts().head(10)
    for opp, count in opp_counts.items():
        avg_so = df[df['Opp'] == opp]['opp_so_avg'].mean()
        print(f"  {opp}: {count} games, average SO% = {avg_so:.2f}%")
    
    # Analysis by pitcher
    print(f"\n⚾ Analysis by pitcher (Top 10 with most games):")
    pitcher_counts = df['Pitcher_Name'].value_counts().head(10)
    for pitcher, count in pitcher_counts.items():
        avg_opp_so = df[df['Pitcher_Name'] == pitcher]['opp_so_avg'].mean()
        print(f"  {pitcher}: {count} games, average opponent SO% = {avg_opp_so:.2f}%")
    
    # Analysis by year
    print(f"\n📅 Analysis by year:")
    for year in sorted(df['Season'].unique()):
        year_data = df[df['Season'] == year]
        print(f"  {year}: {len(year_data)} games, average opponent SO% = {year_data['opp_so_avg'].mean():.2f}%")
    
    # Checking correlation between pitcher SO and opponent SO%
    print(f"\n🔗 Correlation between pitcher SO and opponent SO%:")
    correlation = df['SO'].corr(df['opp_so_avg'])
    print(f"  Correlation: {correlation:.3f}")
    
    if abs(correlation) > 0.1:
        print(f"  💡 There is a {'positive' if correlation > 0 else 'negative'} moderate correlation")
    else:
        print(f"  💡 Very low correlation")

if __name__ == "__main__":
    try:
        # Executing the merge
        merged_data = merge_pitcher_team_strikeout_data()
        
        # Analyzing the results
        analyze_merge_results(merged_data)
        
        print(f"\n✅ Process completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during process: {str(e)}")
        import traceback
        traceback.print_exc()
