# Sistema de Otimização de Estoque

Sistema completo para otimização de estoque de um centro de distribuição que abastece 10 lojas com diferentes padrões de vendas.

## Funcionalidades

### 1. Geração de Dados Realistas
- 10 lojas com perfis distintos (grande, média, pequena)
- 8 produtos com características específicas
- Histórico de vendas com sazonalidade semanal
- Variação de 20-40% entre lojas

### 2. Análise Estatística
- Cálculo de demanda média e desvio padrão
- Análise de padrões de sazonalidade semanal
- Identificação de tendências (crescente/decrescente/estável)
- Classificação ABC-XYZ dos produtos

### 3. Otimização de Estoque
- Cálculo de estoque de segurança (modelo estatístico)
- Determinação do ponto de reorder
- Cálculo de EOQ (Quantidade Econômica de Pedido)
- Recomendações automáticas de reposição priorizadas

### 4. Alertas e Monitoramento
- Alertas de estoque crítico
- Status em tempo real por loja/produto
- Classificação: CRÍTICO, ALERTA, ADEQUADO, EXCESSO

### 5. Simulação What-If
- Simular cenários de aumento/redução de demanda
- Avaliar impacto nos custos
- Prever mudanças de status

### 6. Relatórios e Visualização
- Dashboard visual com status por loja
- Gráficos de tendência de demanda
- Relatório de custos (manutenção vs risco de falta)
- Exportação para CSV e TXT

### 7. Otimização Linear (PuLP)
- **Alocação ótima**: Distribui estoque do CD para lojas minimizando custos
- **Otimização com orçamento**: Decide compras respeitando orçamento limitado
- **Mix de produtos**: Maximiza margem considerando restrições de espaço
- **Planejamento multi-período**: Planeja reposições para múltiplas semanas

### 8. Previsão de Demanda (Exponential Smoothing)
- **Simple Exponential Smoothing (SES)**: Para séries estacionárias
- **Holt's Linear Trend**: Para séries com tendência
- **Holt-Winters**: Para séries com tendência e sazonalidade semanal
- **Seleção automática**: Escolhe o melhor método baseado no MAPE
- **Intervalos de confiança**: Previsões com limites superior e inferior

### 9. Dashboard Web (Streamlit)
- **Design estilo Tableau**: Interface moderna e profissional
- **KPIs em tempo real**: Cards com métricas principais
- **Gráficos interativos**: Plotly com hover e zoom
- **Filtros dinâmicos**: Por loja, produto e status
- **5 abas funcionais**: Visão Geral, Por Loja, Reposição, Previsão, Otimização
- **Exportação de dados**: Download em CSV

## Estrutura do Projeto

```
├── main.py                     # Script principal (CLI)
├── app.py                      # Dashboard Streamlit (Web)
├── requirements.txt            # Dependências do projeto
├── README.md                   # Documentação
├── src/
│   ├── __init__.py
│   ├── data_generator.py       # Geração de dados sintéticos
│   ├── statistical_analysis.py # Análises estatísticas
│   ├── inventory_optimizer.py  # Otimização e recomendações
│   ├── linear_optimization.py  # Otimização linear (PuLP)
│   ├── forecasting.py          # Previsão de demanda (Holt-Winters)
│   └── visualization.py        # Gráficos e relatórios
├── reports/                    # Relatórios gerados
├── data/                       # Dados (se necessário)
└── tests/                      # Testes (futuro)
```

## Instalação

```bash
# Clonar o repositório
git clone <url-do-repositorio>
cd nlp_sistema_extracao_regex_pdf_v1

# Instalar dependências
pip install -r requirements.txt
```

## Uso

### Dashboard Web (Recomendado)
```bash
# Iniciar o dashboard Streamlit
streamlit run app.py

# O navegador abrirá automaticamente em http://localhost:8501
```

### Linha de Comando (CLI)
```bash
python main.py
```

### Opções de Linha de Comando
```bash
# Analisar 60 dias de histórico
python main.py --dias 60

# Simular aumento de 30% na demanda
python main.py --cenario 1.3

# Simular redução de 20% na demanda
python main.py --cenario 0.8

# Salvar relatórios em pasta específica
python main.py --output meus_relatorios

# Definir seed para reprodutibilidade
python main.py --seed 123

# Executar otimização linear de alocação
python main.py --otimizar

# Otimizar compras com orçamento de R$ 50.000
python main.py --orcamento 50000

# Otimizar mix de produtos
python main.py --otimizar-mix

# Combinar múltiplas otimizações
python main.py --otimizar --orcamento 100000

# Gerar previsão de demanda para 7 dias
python main.py --prever 7

# Previsão com método específico (auto, ses, holt, holt_winters)
python main.py --prever 14 --metodo-previsao holt_winters

# Combinação completa
python main.py --otimizar --prever 7 --cenario 1.2
```

### Uso como Biblioteca
```python
from main import SistemaOtimizacaoEstoque

# Inicializar sistema
sistema = SistemaOtimizacaoEstoque(dias_historico=30)
sistema.inicializar()

# Obter resumo
resumo = sistema.obter_dashboard_resumo()
print(f"Itens críticos: {resumo['criticos']}")

# Obter recomendações
recomendacoes = sistema.obter_recomendacoes(top_n=10)
print(recomendacoes)

# Simular cenário
simulacao = sistema.simular_cenario(fator_demanda=1.5)

# Gerar relatórios
arquivos = sistema.gerar_relatorios("reports")
```

## Produtos Monitorados

| ID | Produto | Demanda Base | Tempo Reposição |
|---|---------|--------------|-----------------|
| PROD_001 | Arroz 5kg | 50 un/dia | 2 dias |
| PROD_002 | Feijão 1kg | 80 un/dia | 2 dias |
| PROD_003 | Óleo de Soja 900ml | 60 un/dia | 3 dias |
| PROD_004 | Açúcar 1kg | 70 un/dia | 2 dias |
| PROD_005 | Refrigerante 2L | 120 un/dia | 1 dia |
| PROD_006 | Cerveja Pack 12 | 90 un/dia | 1 dia |
| PROD_007 | Leite 1L | 100 un/dia | 1 dia |
| PROD_008 | Detergente 500ml | 40 un/dia | 4 dias |

## Lojas Monitoradas

| ID | Tipo | Multiplicador de Demanda |
|---|------|-------------------------|
| Loja_01 | Grande | 1.4x |
| Loja_02 | Grande | 1.2x |
| Loja_03 | Média | 1.0x |
| Loja_04 | Média | 0.9x |
| Loja_05 | Média | 1.1x |
| Loja_06 | Pequena | 0.8x |
| Loja_07 | Pequena | 0.7x |
| Loja_08 | Grande | 1.3x |
| Loja_09 | Média | 0.85x |
| Loja_10 | Pequena | 0.75x |

## Modelo Estatístico

### Estoque de Segurança
O sistema utiliza o modelo clássico de estoque de segurança:

```
SS = Z × σ_LT
```

Onde:
- **Z**: Fator de segurança baseado no nível de serviço (95% padrão)
- **σ_LT**: Desvio padrão da demanda durante o lead time
- **σ_LT = σ_d × √(LT)** para demanda com variância constante

### Ponto de Reorder
```
ROP = (Demanda média × Lead Time) + Estoque de Segurança
```

### Quantidade Econômica de Pedido (EOQ)
```
EOQ = √(2 × D × S / H)
```

Onde:
- **D**: Demanda anual
- **S**: Custo por pedido
- **H**: Custo de manutenção por unidade/ano

## Saídas Geradas

### Dashboard (dashboard_estoque.png)
- Status de estoque por loja (barras empilhadas)
- Distribuição de dias de estoque
- Top 10 itens com menor cobertura
- Resumo geral com métricas

### Relatório de Texto (relatorio_estoque.txt)
- Resumo geral do estoque
- Análise de custos por loja
- Lista de itens críticos
- Recomendações de reposição priorizadas
- Itens em excesso de estoque

### Arquivos CSV
- `metricas_estoque.csv`: Métricas detalhadas por loja/produto
- `recomendacoes.csv`: Lista de recomendações de reposição
- `alertas.csv`: Alertas ativos

## Otimização Linear

O sistema utiliza a biblioteca PuLP para resolver problemas de otimização linear.

### Problemas Resolvidos

#### 1. Alocação Ótima de Estoque
Distribui o estoque do Centro de Distribuição para as lojas minimizando:
- Custo de manutenção de estoque
- Risco de stockout (falta de produtos)

```python
# Via código
resultado = sistema.otimizar_alocacao(nivel_servico=0.95)

# Via CLI
python main.py --otimizar
```

#### 2. Otimização com Orçamento Limitado
Decide quanto comprar de cada produto respeitando um orçamento máximo:

```python
# Via código
resultado = sistema.otimizar_compras_orcamento(orcamento=50000)

# Via CLI
python main.py --orcamento 50000
```

#### 3. Otimização de Mix de Produtos
Maximiza a margem de lucro considerando restrições de espaço:

```python
# Via código
resultado = sistema.otimizar_mix_produtos(espaco_total=5000)

# Via CLI
python main.py --otimizar-mix
```

### Formulação Matemática

#### Função Objetivo (Minimização de Custo)
```
Minimizar: Σ (custo_manutencao × estoque) + Σ (custo_stockout × falta)
```

#### Restrições
- Disponibilidade no CD: Σ alocação[loja] ≤ estoque_disponivel
- Nível de serviço: estoque + alocação ≥ nível_serviço × demanda
- Orçamento: Σ (preço × quantidade) ≤ orçamento_total

## Previsão de Demanda

O sistema utiliza a biblioteca statsmodels para previsão de demanda com Exponential Smoothing.

### Métodos Disponíveis

#### 1. Simple Exponential Smoothing (SES)
Para séries estacionárias sem tendência ou sazonalidade:

```python
# Via código
resultado = sistema.prever_demanda(horizonte=7, metodo="ses")

# Via CLI
python main.py --prever 7 --metodo-previsao ses
```

#### 2. Holt's Linear Trend
Para séries com tendência mas sem sazonalidade:

```python
# Via código
resultado = sistema.prever_demanda(horizonte=7, metodo="holt")

# Via CLI
python main.py --prever 7 --metodo-previsao holt
```

#### 3. Holt-Winters (Triple Exponential Smoothing)
Para séries com tendência E sazonalidade (recomendado para dados de vendas):

```python
# Via código
resultado = sistema.prever_demanda(horizonte=7, metodo="holt_winters")

# Via CLI
python main.py --prever 7 --metodo-previsao holt_winters
```

#### 4. Seleção Automática
Testa todos os métodos e escolhe o com menor MAPE (erro percentual):

```python
# Via código (padrão)
resultado = sistema.prever_demanda(horizonte=7, metodo="auto")

# Via CLI
python main.py --prever 7
```

### Métricas de Qualidade

O sistema calcula automaticamente:
- **MAE** (Mean Absolute Error): Erro absoluto médio
- **RMSE** (Root Mean Square Error): Raiz do erro quadrático médio
- **MAPE** (Mean Absolute Percentage Error): Erro percentual médio

## Extensões Futuras

- [x] ~~Otimização usando programação linear~~
- [x] ~~Previsão com suavização exponencial (Holt-Winters)~~
- [x] ~~Interface web com Streamlit~~
- [ ] Integração com banco de dados
- [ ] API REST para integração com ERPs

## Requisitos

- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.23.0
- scipy >= 1.9.0
- matplotlib >= 3.6.0
- pulp >= 2.7.0 (otimização linear)
- statsmodels >= 0.14.0 (previsão de demanda)
- streamlit >= 1.25.0 (dashboard web)
- plotly >= 5.15.0 (gráficos interativos)

## Licença

MIT License
