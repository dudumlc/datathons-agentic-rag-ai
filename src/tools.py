import streamlit as st
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from groq import Groq # Certifique-se de que 'groq' está instalado: pip install groq


# ==============================================================================
# 1. FERRAMENTAS / AÇÕES CRIADAS PARA A IA (Sem alterações)
# ==============================================================================

def calculate(operation: str) -> float:
    return eval(operation)

def get_average_dataframe(df, column):
    # A avaliação do df é feita antes desta chamada no eval.
    # df deve ser um DataFrame resultante de outra função (e.g., get_coff(...)).
    if isinstance(df, pd.DataFrame):
        if column in df.columns:
            # Retorna o item numérico para ser usado como Observation
            return df[column].mean().item()
        else:
            return f"Erro: Coluna '{column}' não encontrada no DataFrame."
    return "Erro: O primeiro argumento não é um DataFrame válido."

def filter_dataframe(df, column, value):
    # df deve ser um DataFrame resultante de outra função.
    if isinstance(df, pd.DataFrame):
        if column in df.columns:
            df_final = df[df[column] == value]
            return df_final
        else:
            return f"Erro: Coluna '{column}' não encontrada no DataFrame."
    return "Erro: O primeiro argumento não é um DataFrame válido."

def get_current_date(period):
    time_data = datetime.datetime.now()
    match period.lower():
        case "year":
            return time_data.year
        case "month":
            return time_data.month
        case "day":
            return time_data.day
        case "hour":
            return time_data.hour
        case "minute":
            return time_data.minute
        case "second":
            return time_data.second
        case 'all':
            return time_data.strftime("%Y-%m-%d %H:%M:%S")
        case _:
            return 'Invalid period'

def get_lineplot(dataframe, date_column, value_column, hue = None):
    # Esta função agora retorna a figura de Matplotlib para o Streamlit
    if isinstance(dataframe, pd.DataFrame):
        if date_column in dataframe.columns and value_column in dataframe.columns:
            plt.figure(figsize=(15,4))
            sns.lineplot(data=dataframe, x=date_column, y=value_column, hue = hue)
            # Retorna a figura, Streamlit usará st.pyplot() para exibi-la
            return plt.gcf() 
        else:
             return f"Erro: Colunas especificadas ('{date_column}' ou '{value_column}') não encontradas."
    return "Erro: O primeiro argumento não é um DataFrame válido para plotagem."

# Funções que acessam o S3 ONS
def get_coff(ano, mes, data_inicio, data_fim):
    try:
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        bucket = "ons-aws-prod-opendata"
        key = f"dataset/restricao_coff_fotovoltaica_detail_tm/RESTRICAO_COFF_FOTOVOLTAICA_DETAIL_{ano}_{mes}.csv"
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=3600,
        )
        df = pd.read_csv(url, sep=";")
        df["din_instante"] = pd.to_datetime(df["din_instante"])
        
        if data_inicio:
            df = df[df["din_instante"] >= pd.to_datetime(data_inicio)]
        if data_fim:
            df = df[df["din_instante"] <= pd.to_datetime(data_fim)]
            
        return df
    except Exception as e:
        return f"Erro ao acessar S3/ONS para COFF: {e}"

def get_geracao_usina(ano, mes, data_inicio, data_fim):
    try:
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        bucket = "ons-aws-prod-opendata"
        key = f"dataset/geracao_usina_2_ho/GERACAO_USINA-2_{ano}_{mes}.csv"
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=3600,
        )
        df = pd.read_csv(url, sep=";")
        df["din_instante"] = pd.to_datetime(df["din_instante"])

        if data_inicio:
            df = df[df["din_instante"] >= pd.to_datetime(data_inicio)]
        if data_fim:
            df = df[df["din_instante"] <= pd.to_datetime(data_fim)]
            
        return df
    except Exception as e:
        return f"Erro ao acessar S3/ONS para Geração Usina: {e}"

def get_programacao_energia_periodo(data_inicio, data_fim):
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    bucket = "ons-aws-prod-opendata"
    datas = pd.date_range(start=data_inicio, end=data_fim, freq="D")
    dfs = []
    
    for data in datas:
        ano = data.strftime("%Y")
        mes = data.strftime("%m")
        dia = data.strftime("%d")
        key = f"dataset/programacao_diaria/PROGRAMACAO_DIARIA_{ano}_{mes}_{dia}.csv"
        
        try:
            url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=3600,
            )
            df = pd.read_csv(url, sep=";", low_memory=False)
            df["din_instante"] = pd.to_datetime(df["din_programacaodia"])
            dfs.append(df)
        except Exception:
            # Em Streamlit, é melhor evitar 'print' e usar 'st.warning' ou 'st.error'
            pass 

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame() # Retorna DataFrame vazio se não encontrar