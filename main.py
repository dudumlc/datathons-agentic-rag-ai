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

from src.tools import *
from src.agent import ReactAgent


# MENSAGEM DE SISTEMA - DEFINIÇÕES DO COMPORTAMENTO DO AGENTE (ESTABELECENDO O FUNCIONAMENTO EM REACT)

system_prompt = """
You run in a loop of Thought, Action, PAUSE, Observation, Answer.
Use Thought to analyze the the question you have been asked and describe all your thoughts about what you need to do to answer correctly.
Use Action to run one of the actions available to you and that you analyzed as you needed - then return PAUSE. (skip this step only if does not have Action needed, and go directly to Answer).
Observation will be the result of running those actions.
At the end of the loop you output an Answer when you have the final answer to the question. After that you are finished, do not output anything else.

Do not make up answers, if you do not know the answer, say you do not know.
Do not write anything outside the Thought, Action, PAUSE, Observation, Answer structure.
Just answer exactly what the user asks. If he asks a plot, just say that the plot is below. If he asks a dataframe, just say that the dataframe is below.

Only answer to the user without Thought, Action, PAUSE, Observation, if the question is about the definition of something. In this cases and only in this cases, just output Answer: your explanation

Your available actions are:

calculate:
e.g. calculate: "4 * 7 / 3"
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary. Right the operations between quotes.

get_average_dataframe:
e.g. get_average_dataframe: get_coff("2025", "08", "2025-09-01", "2025-09-01") , "val_geracao"
returns ONLY the numerical average of a specific column from a dataframe obtained by another tool. In this case, yo need to pass the full call of the other tool as argument. Do not include [get_average_dataframe: args] in the end of your answer. This command is specific to visualizations.
After you get the average value, just pass the value to the user, you do not need to get the dataframe.

filter_dataframe:
e.g. filter_dataframe: get_geracao_usina("2025", "09", "2025-09-01", "2025-09-01") , "nom_tipousina", "FOTOVOLTAICA"
returns a dataframe filtered by a specific column. In this case, yo need to pass the full call of the other tool that returns a dataframe as argument. Do not include [filter_dataframe: args] in the end of your answer. This command is specific to visualizations.
If you want filter get_geracao_usina, you have the options below to each column:
nom_subsistema -> 'NORTE', 'NORDESTE', 'SUL', 'SUDESTE', 'PARAGUAI'
nom_tipousina -> 'FOTOVOLTAICA', 'HIDROELÉTRICA', 'TÉRMICA', 'EOLIELÉTRICA', 'NUCLEAR'
nom_tipocombustivel -> 'Fotovoltaica', 'Hidráulica', 'Gás', 'Óleo Combustível', 'Eólica', 'Carvão', 'Outras Multi-Combustível', 'Resíduos Industriais', 'Biomassa', 'Multi-Combustível Diesel/Óleo', 'Óleo Diesel', 'Nuclear'


get_lineplot:
e.g. get_lineplot: dataframe = get_geracao_usina("2025", "09", "2025-09-01 00:00:00", "2025-09-30 23:00:00"), date_column = "din_instante", value_column = "val_geracao", hue = "nom_tipousina"
returns a lineplot image of a dataframe value column. Analyze the columns available in dataframe, identify the date column, the value columns and the categorical columns which user wants to analyze. If user wants to see a lineplot divided by each category, use hue, otherwise use hue=None.
Do not use it as Action, just use this tool to show the the data of a dataframe that contains the data to be plotted. Include [LINEPLOT: get_lineplot: dataframe = tool(args), date_column = "date_column_name", value_column = "value_column_name", hue = "category_column_to_segregate_visual"] in the end of your final answer if user asked for a line plot.


Use it only when user asks a lineplot, do not use it in other moments

Date column for get_coff: din_instante
Date column for get_geracao_usina: din_instante
Date column for get_programacão_energia_periodo: din_programacaodia


get_current_date:
e.g. get_current_date: "day"
returns the current date decomposed into day, month, year, hour, minute, second. when 'all' is passed, return all componenents

get_coff:
e.g. get_coff: "2025", "09", "2025-09-01 00:00:00", "2025-09-02 23:00:00"
returns a dataframe with the constrained off energy between two dates in a month. Always use format YYYY-MM-DD HH:00:00 to initial date and end date.

get_geracao_usina:
e.g. get_geracao_usina: "2025", "09", "2025-09-01 00:00:00", "2025-09-02 23:00:00"
returns a dataframe with the generation of the usina between two dates in a month. Always use format YYYY-MM-DD HH:00:00 to initial date and end date.

get_programacao_energia_periodo:
e.g. get_programacao_energia_periodo: "2025-09-20", "2025-09-20"
returns a dataframe with the generation of the usina between two dates 

------
Example session:
Question: What is the price of HGLG11 in september 2025?
Thought: I need to find the historical price of HGLG11 for september 2025
Action: get_history_stocks_price: "HGLG11", "2025-09-01", "2025-10-01"
PAUSE

You will be called again with this:
Observation: (returns a dataframe with the historical prices for HGLG11 in september 2025)

If the dataframe generated have the answer, output that you found the result as the Answer and show the action and the arguments used between brackets. You do not need to insert the dataframe, it will be shown to the user as a pandas dataframe visual.

Answer: The price of HGLG11 in september can be found in the dataframe returned below. [DATAFRAME: get_history_stocks_price: "HGLG11", "2025-09-01", "2025-10-01"]

------
Example session 2:
Question: What is the average price of generated energy in september 2025?
Thought: I need to find the generated energy for september 2025
Action: get_geracao_usina: "2025", "09"
PAUSE

You will be called again with this:
Observation: (returns a dataframe with the generated energy in september 2025)

Thought: Now that I have the dataframe containing the generation data for September 2025, I can calculate the average of the 'val_geracao' column, which represents the energy generation values.

Action: get_average_dataframe: get_geracao_usina("2025", "09"), "val_geracao"

PAUSE
Observation: 113.67633804827753
Thought: The average generation of energy in September 2025 has been calculated.

Answer: The average generation of energy in September 2025 is 113.68.

------
Example session 3:
Question: What is the average price of generated photovoltaic energy in september 2025?
Thought: I need to find the generated photovoltaic energy for september 2025
Action: filter_dataframe: get_geracao_usina("2025", "09", "2025-09-01", "2025-09-30"), "nom_tipousina" , "FOTOVOLTAICA"
PAUSE

You will be called again with this:
Observation: (returns a dataframe with the generated photovoltaic energy in september 2025)

Thought: Now that I have the dataframe containing the generated photovoltaic energy data for September 2025, I can calculate the average of the 'val_geracao' column, which represents the energy generation values.

Action: get_average_dataframe: filter_dataframe(get_geracao_usina("2025", "09", "2025-09-01", "2025-09-30"), "nom_tipousina" , "FOTOVOLTAICA")

PAUSE
Observation: 0.00086
Thought: The average generation of photovoltaic energy in September 2025 has been calculated. Now I need to calculate the 

Answer: The average generation of photovoltaic energy in September 2025 is 0.00086.

Now it's your turn:
""".strip()

# Mapeamento de Ferramentas para uso no eval (semelhante ao código original)
TOOL_MAPPING = {
    'calculate': calculate,
    'get_current_date': get_current_date,
    'get_average_dataframe': get_average_dataframe,
    'filter_dataframe': filter_dataframe,
    'get_lineplot': get_lineplot,
    'get_coff': get_coff,
    'get_geracao_usina': get_geracao_usina,
    'get_programacao_energia_periodo': get_programacao_energia_periodo
}

# ==============================================================================
# FUNÇÃO EXECUTORA DO AGENTE PARA STREAMLIT
# ==============================================================================

def agent_executor_streamlit(query, agent_instance):
    """
    Executa a lógica do agente e formata o resultado para exibição no Streamlit.
    """
    next_prompt = query
    resposta_final = ""
    dataframe_resultado = None
    plot_figure = None

    i = 0
    while i < 10:
        i += 1
        
        # ⚠️ Feedback visual da iteração (opcional)
        st.caption(f"Iteração: {i} | Último Prompt: {next_prompt[:50]}...") 

        try:
            result = agent_instance(next_prompt)
        except Exception as e:
            resposta_final = f"Erro fatal na execução do agente: {e}"
            break
        
        # 1. Processamento de Ação (PAUSE/Action)
        if "PAUSE" in result and "Action" in result:
            action_match = re.findall(r"Action: ([a-z_]+): (.+)", result, re.IGNORECASE)

            if action_match:
                chosen_tool, arg = action_match[0]
                
                # Validação de ferramenta (uso de TOOL_MAPPING para segurança/claro)
                if chosen_tool in TOOL_MAPPING:
                    try:
                        # Execução da ferramenta via eval (replicando a lógica original)
                        result_tool = eval(f"{chosen_tool}({arg})")
                        next_prompt = f"Observation: {result_tool}"
                    except Exception as e:
                        # Captura erros na execução das funções Python (ex: argumentos inválidos)
                        next_prompt = f"Observation: Erro ao executar ferramenta {chosen_tool} com args {arg}: {e}"
                else:
                    next_prompt = "Observation: Tool not found"
            else:
                 next_prompt = "Observation: Formato de Action não reconhecido."
            
            continue

        # 2. Processamento de Resposta (Answer)
        if "Answer" in result:
            
            # A. Resposta com LINEPLOT
            if "[LINEPLOT:" in result:
                plot_action_match = re.search(r"(?<=\[LINEPLOT:\s)(.*)(?=\])", result)
                if plot_action_match:
                    plot_action_args = plot_action_match.group(1).strip().split(': ')
                    plot_tool = plot_action_args[0]
                    plot_arg = plot_action_args[1]

                    match = re.search(r"Answer:\s*(.*?)\s*\[LINEPLOT", result, re.DOTALL)
                    if match:
                        resposta_final = match.group(1).strip()
                    
                    # Executa a função de plotagem. Ela deve retornar a figura do Matplotlib
                    if plot_tool in TOOL_MAPPING:
                        plot_figure = eval(f"{plot_tool}({plot_arg})")
                        
                    break

            # B. Resposta com DATAFRAME
            elif "[DATAFRAME:" in result:
                dataframe_action_match = re.search(r"(?<=\[DATAFRAME:\s)(.*)(?=\])", result)
                if dataframe_action_match:
                    dataframe_action_args = dataframe_action_match.group(1).strip().split(': ')
                    dataframe_tool = dataframe_action_args[0]
                    dataframe_arg = dataframe_action_args[1]
                    
                    match = re.search(r"Answer:\s*(.*?)\s*\[DATAFRAME", result, re.DOTALL)
                    if match:
                        resposta_final = match.group(1).strip()
                        
                    # Executa a ferramenta e armazena o DataFrame
                    if dataframe_tool in TOOL_MAPPING:
                        dataframe_resultado = eval(f"{dataframe_tool}({dataframe_arg})")
                        
                    break

            # C. Resposta Simples
            match = re.search(r"Answer:\s*(.*)", result, re.DOTALL)
            if match:
                resposta_final = match.group(1).strip()
                break
        
        if i == 10:
             resposta_final = "Limite de 10 iterações atingido sem uma resposta final."

    return resposta_final, dataframe_resultado, plot_figure


# ==============================================================================
# INTERFACE STREAMLIT
# ==============================================================================

st.set_page_config(page_title="Agente ReAct ONS", layout="wide")
st.title("⚡ FairWatt - Agente de Dados Abertos ONS")

# Se for a primeira execução ou se a chave API não estiver no estado da sessão
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ''
    
# Campo para API Key e Histórico de Chat
with st.sidebar:
    st.header("Configuração")
    st.session_state.groq_api_key = st.text_input(
        "Sua API Key da Groq", 
        type="password", 
        value=st.session_state.groq_api_key, 
        help="Necessário para instanciar o agente."
    )
    st.markdown("---")
    st.markdown("⚠️ O código utiliza `boto3` para acesso público (UNSIGNED) a dados ONS no S3. Não requer credenciais AWS.")


# Inicializa o histórico de chat E o Agente
if "messages" not in st.session_state:
    st.session_state.messages = []

# Instancia o agente (ocorre a cada re-run do Streamlit)
agent_instance = ReactAgent(
    system=system_prompt,
    api_key=st.session_state.groq_api_key,
    tools=list(TOOL_MAPPING.keys()),
)

# Exibe o histórico de chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("dataframe") is not None:
            st.dataframe(message["dataframe"])
        if message.get("plot_figure") is not None:
             st.pyplot(message["plot_figure"])

# Campo de entrada do usuário
if prompt := st.chat_input("Faça sua pergunta (e.g., 'Qual a média de geração de usina em 2024?'):"):
    # 1. Adiciona a pergunta do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Executa o Agente
    with st.chat_message("assistant"):
        with st.spinner("O Agente está pensando e executando as ações..."):
            resposta, df, plot_fig = agent_executor_streamlit(prompt, agent_instance)
            
            # Exibe a resposta de texto
            st.markdown(resposta)
            
            # Exibe o DataFrame, se houver
            if df is not None and isinstance(df, pd.DataFrame):
                st.dataframe(df)
            
            # Exibe o Plot, se houver
            if plot_fig is not None:
                st.pyplot(plot_fig)
                plt.close(plot_fig) # Limpa a figura do Matplotlib

    # 3. Adiciona a resposta final ao histórico (para persistir no re-run)
    assistant_message = {
        "role": "assistant",
        "content": resposta,
        # Armazena DataFrame e Figura de Plot para reexibição
        "dataframe": df if df is not None and isinstance(df, pd.DataFrame) else None,
        "plot_figure": plot_fig
    }
    st.session_state.messages.append(assistant_message)