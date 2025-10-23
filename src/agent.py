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
# 2. DESENVOLVIMENTO DO AGENTE EM FORMATO REACT (Com ajustes para Streamlit)
# ==============================================================================

class ReactAgent:
    def __init__(self, system: str = "", api_key: str = '', tools: list = []) -> None:
        # Use o st.secrets ou um campo de entrada para API key em apps reais
        if not api_key or api_key == 'GROQ_API_KEY':
            st.error("ERRO: A API Key da Groq não está configurada corretamente.")
            self.client = None # Impede a chamada da API
        else:
            self.client = Groq(api_key=api_key)
        
        self.system = system
        self.tools = tools
        self.messages: list = []

        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message=""):
        if not self.client:
             return "Answer: Erro de configuração do Agente: API Key da Groq ausente ou incorreta."
             
        # Garante que a mensagem do usuário/Observation seja tratada como 'user'
        if message:
            # Evita adicionar Observações repetidamente se o fluxo for interrompido
            if not message.startswith("Observation:"):
                 self.messages.append({"role": "user", "content": message})
            else:
                 self.messages.append({"role": "user", "content": message}) # Adiciona Observation como input

        result = self.execute()
        # Adiciona a resposta do assistente ao histórico da classe
        self.messages.append({"role": "assistant", "content": result}) 
        return result

    def execute(self):
        try:
            completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile", # Mantenha o modelo original
                messages=self.messages
            )
            return completion.choices[0].message.content
        except Exception as e:
            # Captura erros da API (e.g., chave inválida, limite)
            return f"Answer: Erro na chamada da API Groq: {e}"
