import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

# Lê seus CSVs
pitchers_data = pd.read_csv('pitchers_data.csv')
predicted_data = pd.read_csv('predicted_2025-04-17.csv')

# Converte os dados para texto
pitchers_text = pitchers_data.to_string(index=False)
predicted_text = predicted_data.to_string(index=False)

# Monta o prompt completo
prompt = f"""
Você é um especialista em análise de baseball e apostas esportivas.

Tenho duas bases de dados:
1. Histórico real de performances de pitchers em 2023 e 2024:
{pitchers_text}

2. Predições de strikeouts para partidas futuras com odds e recomendações:

Hoje buscamos dados de duas fontes:
1. API de odds de apostas, que tem seu proprio modelo de machine learning
2. Modelo de machine learning, que foi treinado com dados de 2023 e 2024 localmente

As colunas são:
Player: Nome do jogador
Name_abbreviation: Abreviação do nome do jogador 
Team: Time do jogador 
Over Line: Linha para a aposta "over" da casa de aposta 
Over Odds: Odds para a aposta "over" da casa de aposta 
Under Line: Linha para a aposta "under" da casa de aposta 
Under Odds: Odds para a aposta "under" da casa de aposta 
API Projected Value: Valor projetado pela API da quantidade de strikeouts 
API Recommended Side: Lado recomendado pela API da quantidade de strikeouts 
Streak: Sequência atual 
Streak Type: Tipo de sequência
Diff: Diferença entre o valor projetado e a linha
ML Strikeout Line: Linha de strikeouts prevista pelo modelo de machine learning
ML Predict Value: Valor previsto pelo modelo de machine learning
ML Recommend Side: Lado recomendado pelo modelo de machine learning
ML Confidence Percentage: Percentual de confiança do modelo de machine learning
Pitcher 2023: Indica se o arremessador jogou em 2023

{predicted_text}

Baseado nesses dados:
- Me dê insights que possam aumentar minhas chances em apostas.
- Considere padrões, anomalias e oportunidades escondidas.
- Aponte jogadores subestimados ou superestimados.
- O principal: faça um ranking das melhores apostas, com informacao de linha, do api predict value e do ml predict value, com a linha de strikeouts e o valor de strikeouts.
- Não precisa colocar o nome abreviado do jogador, apenas o nome e o time.
Por favor, responda de forma detalhada.
"""

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("WARNING: OPENAI_API_KEY is not set in the environment. Please set it before running the application.")
    exit()


# Envia para a OpenAI
client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0.2  # Controla a criatividade da resposta
)

# Exibe o resultado
print(response.choices[0].message.content)