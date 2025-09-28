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


# MENSAGEM DE SISTEMA (Inalterada)
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
# Usei '...' para encurtar a visualização, mas no código real ela deve estar completa.

# Mapeamento de Ferramentas para uso no eval (semelhante ao código original)
TOOL_MAPPING = {
    'calculate': calculate,
    'get_current_date': get_current_date,
    'get_average_dataframe': get_average_dataframe,
    'filter_dataframe': filter_dataframe,
    'get_lineplot': get_lineplot,
    'get_coff': get_coff,
    'get_geracao_usina': get_geracao_usina,
    'get_programacao_energia_periodo': get_programacao_energia_periodo,
    # Você não incluiu 'get_history_stocks_price' no seu código, removi do mapping
}

# ==============================================================================
# 3. FUNÇÃO EXECUTORA DO AGENTE PARA STREAMLIT
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
# 4. INTERFACE STREAMLIT
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