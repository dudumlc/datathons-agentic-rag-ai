# Datathon ONS 2025 - Grupo 13

Bem-vindo ao repositório do Grupo 13 para o Datathon ONS 2025!

## Sobre o Desafio

O Datathon ONS lançou o desafio “Transformando dados abertos em mais valor para o setor elétrico”, com foco no uso de Inteligência Artificial para explorar os datasets disponíveis no Portal de Dados Abertos do ONS
.

O objetivo foi propor soluções que tornem os dados do Setor Elétrico Brasileiro (SEB) mais acessíveis, intuitivos e úteis para diferentes públicos — desde empresas até a sociedade em geral. A iniciativa buscou estimular a transparência, apoiar melhores decisões e agregar valor ao portal por meio de experiências interativas e inteligentes.

## Sobre a Solução

Nosso grupo desenvolveu um agente de IA baseado no framework ReAct (Reasoning + Action), projetado para interagir de forma inteligente com os dados do setor elétrico. Para isso, definimos um conjunto de ferramentas (tools) que permitem:

- Geração de visuais dinâmicos, como gráficos e tabelas;
- Execução de cálculos matemáticos;
- Respostas a perguntas conceituais sobre o setor;
- Entre outras funcionalidades que ampliam a exploração dos dados.

A solução foi integrada diretamente às bases do Portal de Dados Abertos do ONS, acessadas via Amazon S3, garantindo atualização e confiabilidade. Trabalhamos com três conjuntos principais:

- Geração por Usina em Base Horária;
- Valores da Programação Diária;
- Constrained-off por usinas fotovoltaicas.

Com isso, o agente se torna capaz de transformar dados complexos em informações acessíveis e úteis, alinhando-se ao objetivo do desafio de tornar o portal mais intuitivo, interativo e produtivo.

Para tornar o acesso simples e intuitivo, disponibilizamos a solução em uma interface construída com Streamlit, onde o usuário pode explorar os dados e interagir com o agente. Para isso, basta fornecer uma chave de API do GROQ (https://console.groq.com/home).  

## Estrutura do Repositório

A estrutura de pastas deste repositório foi organizada para manter o projeto limpo e modular. Cada diretório principal contém um arquivo `README.md` que detalha seu propósito específico.

- `data/`: Contém os datasets brutos, processados e externos.
- `docs/`: Documentação do projeto, relatórios e apresentações.
- `notebooks/`: Notebooks Jupyter para exploração de dados, modelagem e análise.
- `src/`: Código fonte, scripts e módulos reutilizáveis.
- `results/`: Resultados finais, como submissões, visualizações e modelos treinados.

## Como Começar

1.  **Clone o repositório:**
    ```bash
    git clone git@github.com:DatathONS2025/grupo13.git
    
    ```

2.  **Restaure as dependências:**
    Por exemplo:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Execute o projeto:**
    Por exemplo:
    ```bash
    python main.py
    ```


## Contribuidores

- Luís Eduardo Moreira Las Casas ([Linkedin](https://www.linkedin.com/in/luis-las-casas/))
- Arthur de Albuquerque Santana ([Linkedin](https://www.linkedin.com/in/arthur-albuquerque-santana/))
- Nathália de Castro Nascimento ([Linkedin](https://www.linkedin.com/in/nath%C3%A1lia-nascimento-3617b1293/))
- Felipe de Melo Silva ([Linkedin](https://www.linkedin.com/in/felipe-melo-705051230))
