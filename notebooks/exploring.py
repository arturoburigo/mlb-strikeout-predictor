import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

df = pd.read_csv('../pitchers_data_with_opp_so.csv')


# Exibindo informações básicas
print("=== INFORMAÇÕES BÁSICAS ===")
print(f"Dimensões do dataset: {df.shape}")
print(f"\nPrimeiras 5 linhas:")
print(df.head())

print(f"\nInformações do dataset:")
print(df.info())

print(f"\nEstatísticas descritivas:")
print(df.describe())

# =============================================================================
# CÉLULA 3: VERIFICAÇÃO DE QUALIDADE DOS DADOS
# =============================================================================

# Verificando se há valores nulos
print("=== VERIFICAÇÃO DE DADOS NULOS ===")
print(df.isnull().sum())

# Verificando valores únicos por coluna
print(f"\n=== VALORES ÚNICOS ===")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} valores únicos")

# Verificando se há valores duplicados
print(f"\nLinhas duplicadas: {df.duplicated().sum()}")


df = df[df['IP'] >= 1]

print(f"\nApós remover linhas com IP < 1, o dataset possui {df.shape[0]} linhas e {df.shape[1]} colunas.")

# =============================================================================
# ANÁLISE DE CORRELAÇÃO COM SO (STRIKEOUTS)
# =============================================================================

print("\n" + "="*60)
print("ANÁLISE DE CORRELAÇÃO COM SO (STRIKEOUTS)")
print("="*60)

# Selecionando apenas colunas numéricas para correlação
numeric_columns = df.select_dtypes(include=[np.number]).columns
correlation_with_so = df[numeric_columns].corr()['SO'].sort_values(ascending=False)

print("\n=== CORRELAÇÕES COM SO (ordenadas por magnitude) ===")
for feature, corr in correlation_with_so.items():
    if feature != 'SO':
        print(f"{feature:15} : {corr:8.4f}")

# Criando visualizações
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Análise de Correlação com Strikeouts (SO)', fontsize=16, fontweight='bold')

# 1. Heatmap de correlação
correlation_matrix = df[numeric_columns].corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=axes[0,0])
axes[0,0].set_title('Matriz de Correlação Completa')

# 2. Correlações com SO em barras
so_correlations = correlation_with_so.drop('SO')
colors = ['red' if x < 0 else 'blue' for x in so_correlations.values]
axes[0,1].barh(so_correlations.index, so_correlations.values, color=colors, alpha=0.7)
axes[0,1].set_title('Correlações com SO')
axes[0,1].set_xlabel('Coeficiente de Correlação')
axes[0,1].axvline(x=0, color='black', linestyle='-', alpha=0.3)

# 3. Scatter plot das 3 features mais correlacionadas positivamente
top_positive = so_correlations.head(3)
for i, (feature, corr) in enumerate(top_positive.items()):
    axes[1,0].scatter(df[feature], df['SO'], alpha=0.6, label=f'{feature} (r={corr:.3f})')
axes[1,0].set_xlabel('Valor da Feature')
axes[1,0].set_ylabel('Strikeouts (SO)')
axes[1,0].set_title('Top 3 Features Positivamente Correlacionadas')
axes[1,0].legend()

# 4. Scatter plot das 3 features mais correlacionadas negativamente
top_negative = so_correlations.tail(3)
for i, (feature, corr) in enumerate(top_negative.items()):
    axes[1,1].scatter(df[feature], df['SO'], alpha=0.6, label=f'{feature} (r={corr:.3f})')
axes[1,1].set_xlabel('Valor da Feature')
axes[1,1].set_ylabel('Strikeouts (SO)')
axes[1,1].set_title('Top 3 Features Negativamente Correlacionadas')
axes[1,1].legend()

plt.tight_layout()
plt.savefig('../correlation_analysis_so.png', dpi=300, bbox_inches='tight')
plt.show()

# Análise estatística adicional
print("\n=== ANÁLISE ESTATÍSTICA DETALHADA ===")
print(f"Correlação mais forte positiva: {top_positive.index[0]} (r = {top_positive.iloc[0]:.4f})")
print(f"Correlação mais forte negativa: {top_negative.index[-1]} (r = {top_negative.iloc[-1]:.4f})")

# Correlações significativas (|r| > 0.3)
significant_correlations = so_correlations[abs(so_correlations) > 0.3]
print(f"\nCorrelações significativas (|r| > 0.3):")
for feature, corr in significant_correlations.items():
    strength = "Forte" if abs(corr) > 0.7 else "Moderada" if abs(corr) > 0.5 else "Fraca"
    direction = "Positiva" if corr > 0 else "Negativa"
    print(f"  {feature:15} : {corr:8.4f} ({strength} {direction})")

# ============================================================================
# 2. REMOÇÃO DE FEATURES PROBLEMÁTICAS
# ============================================================================

def remove_problematic_features(df):
    """
    Remove features com correlações negativas/irrelevantes
    """
    print("\n🗑️ REMOVENDO FEATURES PROBLEMÁTICAS")
    print("-" * 50)
    
    # Features para remover (baseado na análise de correlação)
    features_to_remove = [
        'ERA',      # -0.0911 (negativa fraca)
        'FIP',      # -0.1407 (negativa, pode confundir modelo)
        'Home',     # -0.0114 (irrelevante)
        'DR',       # -0.0659 (negativa, provavelmente ruído)
        'aLI',      # -0.0690 (negativa, situações de pressão podem reduzir SO)
        'H',        # 0.0005 (muito fraca + pode ser derivada de outras)
        'BB',       # -0.0529 (fraca negativa)
        'Season'    # 0.1455 (apenas tendência temporal, não preditiva)
    ]
    
    # Verificar quais existem no DataFrame
    existing_features = [f for f in features_to_remove if f in df.columns]
    missing_features = [f for f in features_to_remove if f not in df.columns]
    
    print(f"Features a remover: {existing_features}")
    
    # Remover as features
    df = df.drop(columns=existing_features)
    
    return df

df = remove_problematic_features(df)

print(df.columns)

df.to_csv('../pitchers_data_with_opp_so_cleaned.csv', index=False)