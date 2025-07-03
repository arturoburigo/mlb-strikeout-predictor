# 🔄 Fluxo do Pipeline de Machine Learning - Previsões MLB

## 📋 Visão Geral do Sistema

Este é um sistema automatizado de machine learning que prevê strikeouts de arremessadores da MLB usando dados históricos, estatísticas de equipes e odds de apostas.

---

## 🕐 Cronograma Automatizado

```
🕐 19:10 ET - Pipeline de Dados
🕐 22:40 ET - Email de Previsões  
🕐 11:42 ET - Email de Resultados
🕐 12:00 ET - Upload AWS
🕐 13:00 ET - Limpeza de Arquivos
🕐 22:45 ET - Notificação Telegram
```

---

## 🔄 Fluxo Principal do Pipeline

### 1️⃣ **Coleta de Dados** (19:10 ET)
```
┌─────────────────────────────────────────────────────────────┐
│                    COLETA DE DADOS                          │
├─────────────────────────────────────────────────────────────┤
│ 📊 betting_odds_today.py                                    │
│    └─ Scraping de odds de apostas do dia                    │
│                                                             │
│ 🏟️ get_pitcher_lastseason.py                               │
│    └─ Dados históricos dos arremessadores                   │
│                                                             │
│ 📈 get_pitcher_lastgame.py                                  │
│    └─ Resultados dos jogos anteriores                       │
└─────────────────────────────────────────────────────────────┘
```

### 2️⃣ **Processamento de Dados** 
```
┌─────────────────────────────────────────────────────────────┐
│                PROCESSAMENTO DE DADOS                       │
├─────────────────────────────────────────────────────────────┤
│ 📁 data_utils.py                                            │
│    ├─ Carrega dados dos CSVs                                │
│    ├─ Merge de datasets                                     │
│    ├─ Cálculo de métricas (SO_per_IP, BB_per_IP, K-BB%)    │
│    └─ Tratamento de valores nulos/infinitos                │
│                                                             │
│ 🔧 feature_engineering.py                                   │
│    ├─ Cálculo de performance ponderada                      │
│    ├─ Médias móveis (5 e 10 jogos)                         │
│    ├─ Splits casa/fora                                      │
│    └─ Engenharia de features avançadas                     │
└─────────────────────────────────────────────────────────────┘
```

### 3️⃣ **Treinamento do Modelo**
```
┌─────────────────────────────────────────────────────────────┐
│                 TREINAMENTO DO MODELO                       │
├─────────────────────────────────────────────────────────────┤
│ 🤖 model_training.py                                        │
│    ├─ Random Forest                                         │
│    ├─ XGBoost                                              │
│    ├─ LightGBM                                             │
│    ├─ Gradient Boosting                                     │
│    ├─ Cross-validation (5-fold)                            │
│    ├─ Seleção do melhor modelo                             │
│    └─ Avaliação de performance                             │
└─────────────────────────────────────────────────────────────┘
```

### 4️⃣ **Geração de Previsões**
```
┌─────────────────────────────────────────────────────────────┐
│                GERAÇÃO DE PREVISÕES                         │
├─────────────────────────────────────────────────────────────┤
│ 🎯 predictions.py                                           │
│    ├─ Predição de strikeouts por arremessador              │
│    ├─ Cálculo de confiança                                  │
│    ├─ Recomendação Over/Under                              │
│    ├─ Filtragem de picks de qualidade                      │
│    └─ Geração do arquivo predicted_YYYY-MM-DD.csv          │
└─────────────────────────────────────────────────────────────┘
```

---

## 📧 Sistema de Notificações

### 5️⃣ **Email de Previsões** (22:40 ET)
```
┌─────────────────────────────────────────────────────────────┐
│                 EMAIL DE PREVISÕES                          │
├─────────────────────────────────────────────────────────────┤
│ 📧 email_ml_predictions.py                                  │
│    ├─ Leitura do arquivo predicted_YYYY-MM-DD.csv          │
│    ├─ Formatação em HTML                                    │
│    ├─ Top picks destacados                                  │
│    ├─ Métricas de confiança                                 │
│    └─ Envio via SMTP                                        │
└─────────────────────────────────────────────────────────────┘
```

### 6️⃣ **Email de Resultados** (11:42 ET)
```
┌─────────────────────────────────────────────────────────────┐
│                 EMAIL DE RESULTADOS                         │
├─────────────────────────────────────────────────────────────┤
│ 📊 email_ml_results.py                                      │
│    ├─ Comparação previsões vs resultados                   │
│    ├─ Análise de performance                               │
│    ├─ Estatísticas de acerto                               │
│    ├─ ROI das apostas                                      │
│    └─ Relatório detalhado                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Serviços Auxiliares

### 7️⃣ **Upload AWS** (12:00 ET)
```
┌─────────────────────────────────────────────────────────────┐
│                    UPLOAD AWS                               │
├─────────────────────────────────────────────────────────────┤
│ ☁️ aws_upload.py                                            │
│    ├─ Upload de resultados para S3                          │
│    ├─ Backup de dados históricos                            │
│    ├─ Organização por data                                  │
│    └─ Logs de upload                                        │
└─────────────────────────────────────────────────────────────┘
```

### 8️⃣ **Limpeza de Arquivos** (13:00 ET)
```
┌─────────────────────────────────────────────────────────────┐
│                LIMPEZA DE ARQUIVOS                          │
├─────────────────────────────────────────────────────────────┤
│ 🧹 cleanup_files.py                                         │
│    ├─ Remoção de CSVs antigos                              │
│    ├─ Manutenção de espaço em disco                        │
│    ├─ Preservação de dados importantes                     │
│    └─ Logs de limpeza                                       │
└─────────────────────────────────────────────────────────────┘
```

### 9️⃣ **Telegram Bot** (22:45 ET)
```
┌─────────────────────────────────────────────────────────────┐
│                   TELEGRAM BOT                              │
├─────────────────────────────────────────────────────────────┤
│ 🤖 telegramSender.py                                        │
│    ├─ Leitura das previsões                                │
│    ├─ Formatação para Telegram                              │
│    ├─ Envio de mensagens                                    │
│    ├─ Interação com usuários                                │
│    └─ Logs de envio                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Features do Modelo

### 🎯 **Features Principais**
- **IP**: Innings pitched
- **H**: Hits allowed  
- **BB**: Walks
- **ERA**: Earned run average
- **FIP**: Fielding independent pitching
- **SO_per_IP**: Strikeouts por inning
- **BB_per_IP**: Walks por inning
- **K-BB%**: Diferença entre K e BB por inning

### 📈 **Features Avançadas**
- **SO_rolling_5**: Média móvel de 5 jogos
- **SO_rolling_10**: Média móvel de 10 jogos
- **Home/Away splits**: Performance casa vs fora
- **Team_K%**: % de strikeouts da equipe
- **Opp_K%**: % de strikeouts do oponente

---

## 🔄 Fluxo de Dados Completo

```
📊 Dados Brutos
    ↓
🔧 Engenharia de Features
    ↓
🤖 Treinamento do Modelo
    ↓
🎯 Geração de Previsões
    ↓
📧 Notificações (Email + Telegram)
    ↓
📈 Análise de Resultados
    ↓
☁️ Backup AWS
    ↓
🧹 Limpeza
```

---

## 🛠️ Tecnologias Utilizadas

- **Python**: Linguagem principal
- **Pandas**: Manipulação de dados
- **Scikit-learn**: Machine learning
- **XGBoost/LightGBM**: Modelos avançados
- **APScheduler**: Agendamento de tarefas
- **Selenium**: Web scraping
- **SMTP**: Envio de emails
- **AWS S3**: Armazenamento em nuvem
- **Telegram Bot API**: Notificações

---

## 📈 Métricas de Performance

- **R² Score**: Medida de qualidade do modelo
- **MAE**: Erro absoluto médio
- **Cross-validation**: Validação robusta
- **Confidence Score**: Nível de confiança das previsões
- **ROI**: Retorno sobre investimento das apostas

---

## 🔍 Monitoramento

- **Logs detalhados**: pipeline.log
- **Métricas de performance**: Avaliação contínua
- **Alertas de erro**: Notificações em caso de falha
- **Backup automático**: Preservação de dados
- **Limpeza programada**: Manutenção do sistema 