#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para fazer merge das tabelas pitchers_data.csv e team_strikeout_percentage.csv
Adiciona a coluna opp_so_avg baseada na porcentagem de strikeout do time adversário
"""

import pandas as pd
import numpy as np
from pathlib import Path

def merge_pitcher_team_strikeout_data():
    """
    Faz o merge das tabelas de dados de pitchers com a porcentagem de strikeout por equipe
    """
    
    print("🔄 Iniciando merge das tabelas...")
    
    # Carregando os dados
    print("📊 Carregando dados dos pitchers...")
    pitchers_df = pd.read_csv('../../pitchers_data.csv')
    
    print("📊 Carregando dados de porcentagem de strikeout por equipe...")
    team_so_df = pd.read_csv('../../team_strikeout_percentage.csv')
    
    # Exibindo informações básicas
    print(f"\n📋 Informações dos dados:")
    print(f"Pitchers data: {pitchers_df.shape[0]} linhas, {pitchers_df.shape[1]} colunas")
    print(f"Team strikeout data: {team_so_df.shape[0]} linhas, {team_so_df.shape[1]} colunas")
    
    print(f"\n🔍 Colunas dos dados de pitchers:")
    print(pitchers_df.columns.tolist())
    
    print(f"\n🔍 Colunas dos dados de strikeout por equipe:")
    print(team_so_df.columns.tolist())
    
    # Verificando valores únicos nas colunas de join
    print(f"\n📊 Valores únicos em 'Season' (pitchers): {sorted(pitchers_df['Season'].unique())}")
    print(f"📊 Valores únicos em 'Opp' (pitchers): {sorted(pitchers_df['Opp'].unique())}")
    print(f"📊 Valores únicos em 'Team' (strikeout): {sorted(team_so_df['Team'].unique())}")
    
    # Preparando o dataframe de strikeout para o merge
    print("\n🔄 Preparando dados de strikeout para merge...")
    
    # Transformando o dataframe de strikeout de wide para long format
    team_so_long = team_so_df.melt(
        id_vars=['Team'], 
        value_vars=['2023', '2024', '2025'], 
        var_name='Season', 
        value_name='opp_so_avg'
    )
    
    # Formatando a coluna opp_so_avg para duas casas decimais
    team_so_long['opp_so_avg'] = team_so_long['opp_so_avg'].round(2)
    
    # Convertendo Season para string para garantir compatibilidade
    team_so_long['Season'] = team_so_long['Season'].astype(str)
    pitchers_df['Season'] = pitchers_df['Season'].astype(str)
    
    print(f"📊 Team strikeout data após transformação: {team_so_long.shape[0]} linhas")
    print(f"📊 Exemplo dos dados transformados:")
    print(team_so_long.head(10))
    
    # Fazendo o merge
    print("\n🔗 Fazendo o merge das tabelas...")
    
    # Merge baseado em Season e Opp (time adversário)
    merged_df = pitchers_df.merge(
        team_so_long,
        left_on=['Season', 'Opp'],
        right_on=['Season', 'Team'],
        how='left'
    )
    
    # Removendo a coluna Team duplicada (se existir)
    if 'Team' in merged_df.columns:
        merged_df = merged_df.drop('Team', axis=1)
    
    # Verificando o resultado do merge
    print(f"\n✅ Merge concluído!")
    print(f"📊 Dados originais: {pitchers_df.shape[0]} linhas")
    print(f"📊 Dados após merge: {merged_df.shape[0]} linhas")
    print(f"📊 Colunas após merge: {merged_df.shape[1]} colunas")
    
    # Verificando valores nulos na nova coluna
    null_count = merged_df['opp_so_avg'].isnull().sum()
    print(f"📊 Valores nulos em 'opp_so_avg': {null_count} ({null_count/len(merged_df)*100:.2f}%)")
    
    if null_count > 0:
        print("\n⚠️  Valores nulos encontrados. Verificando possíveis causas...")
        
        # Verificando quais times não foram encontrados
        null_opps = merged_df[merged_df['opp_so_avg'].isnull()]['Opp'].unique()
        print(f"📊 Times não encontrados no merge: {sorted(null_opps)}")
        
        # Verificando se há diferenças nos nomes dos times
        pitchers_opps = set(pitchers_df['Opp'].unique())
        team_so_teams = set(team_so_df['Team'].unique())
        
        missing_in_team_so = pitchers_opps - team_so_teams
        missing_in_pitchers = team_so_teams - pitchers_opps
        
        if missing_in_team_so:
            print(f"📊 Times em pitchers_data.csv que não estão em team_strikeout_percentage.csv: {sorted(missing_in_team_so)}")
        
        if missing_in_pitchers:
            print(f"📊 Times em team_strikeout_percentage.csv que não estão em pitchers_data.csv: {sorted(missing_in_pitchers)}")
    
    # Exibindo estatísticas da nova coluna
    print(f"\n📊 Estatísticas da coluna 'opp_so_avg':")
    print(merged_df['opp_so_avg'].describe())
    
    # Exibindo algumas linhas de exemplo
    print(f"\n📊 Exemplo dos dados após merge:")
    print(merged_df[['Season', 'Pitcher', 'Opp', 'opp_so_avg']].head(10))
    
    # Salvando o resultado
    output_file = '../pitchers_data_with_opp_so.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"\n💾 Dados salvos em: {output_file}")
    
    # Verificando a qualidade do merge
    print(f"\n🔍 Verificação da qualidade do merge:")
    
    # Verificando se todos os anos foram processados
    years_processed = merged_df['Season'].unique()
    print(f"📊 Anos processados: {sorted(years_processed)}")
    
    # Verificando a distribuição da nova coluna por ano
    print(f"\n📊 Distribuição de 'opp_so_avg' por ano:")
    for year in sorted(years_processed):
        year_data = merged_df[merged_df['Season'] == year]['opp_so_avg']
        print(f"  {year}: Média = {year_data.mean():.2f}%, Mediana = {year_data.median():.2f}%")
    
    # Verificando se há valores extremos ou suspeitos
    print(f"\n📊 Verificação de valores extremos:")
    print(f"  Mínimo: {merged_df['opp_so_avg'].min():.2f}%")
    print(f"  Máximo: {merged_df['opp_so_avg'].max():.2f}%")
    
    # Verificando se os valores fazem sentido (entre 50% e 80%)
    valid_range = merged_df[(merged_df['opp_so_avg'] >= 50) & (merged_df['opp_so_avg'] <= 80)]
    print(f"  Valores no range esperado (50-80%): {len(valid_range)}/{len(merged_df)} ({len(valid_range)/len(merged_df)*100:.2f}%)")
    
    return merged_df

def analyze_merge_results(df):
    """
    Analisa os resultados do merge
    """
    print(f"\n📈 ANÁLISE DOS RESULTADOS DO MERGE")
    print("=" * 50)
    
    # Análise por time adversário
    print(f"\n🏟️  Análise por time adversário (Top 10 com mais jogos):")
    opp_counts = df['Opp'].value_counts().head(10)
    for opp, count in opp_counts.items():
        avg_so = df[df['Opp'] == opp]['opp_so_avg'].mean()
        print(f"  {opp}: {count} jogos, SO% médio = {avg_so:.2f}%")
    
    # Análise por pitcher
    print(f"\n⚾ Análise por pitcher (Top 10 com mais jogos):")
    pitcher_counts = df['Pitcher'].value_counts().head(10)
    for pitcher, count in pitcher_counts.items():
        avg_opp_so = df[df['Pitcher'] == pitcher]['opp_so_avg'].mean()
        print(f"  {pitcher}: {count} jogos, SO% médio dos adversários = {avg_opp_so:.2f}%")
    
    # Análise por ano
    print(f"\n📅 Análise por ano:")
    for year in sorted(df['Season'].unique()):
        year_data = df[df['Season'] == year]
        print(f"  {year}: {len(year_data)} jogos, SO% médio dos adversários = {year_data['opp_so_avg'].mean():.2f}%")
    
    # Verificando correlação entre SO do pitcher e SO% do adversário
    print(f"\n🔗 Correlação entre SO do pitcher e SO% do adversário:")
    correlation = df['SO'].corr(df['opp_so_avg'])
    print(f"  Correlação: {correlation:.3f}")
    
    if abs(correlation) > 0.1:
        print(f"  💡 Há uma correlação {'positiva' if correlation > 0 else 'negativa'} moderada")
    else:
        print(f"  💡 Correlação muito baixa")

if __name__ == "__main__":
    try:
        # Executando o merge
        merged_data = merge_pitcher_team_strikeout_data()
        
        # Analisando os resultados
        analyze_merge_results(merged_data)
        
        print(f"\n✅ Processo concluído com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro durante o processo: {str(e)}")
        import traceback
        traceback.print_exc()
