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

## Estrutura do Projeto

```
├── main.py                     # Script principal de execução
├── requirements.txt            # Dependências do projeto
├── README.md                   # Documentação
├── src/
│   ├── __init__.py
│   ├── data_generator.py       # Geração de dados sintéticos
│   ├── statistical_analysis.py # Análises estatísticas
│   ├── inventory_optimizer.py  # Otimização e recomendações
│   ├── linear_optimization.py  # Otimização linear (PuLP)
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

### Execução Básica
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

## Extensões Futuras

- [x] ~~Otimização usando programação linear~~
- [ ] Previsão com suavização exponencial (Holt-Winters)
- [ ] Interface web com Streamlit
- [ ] Integração com banco de dados
- [ ] API REST para integração com ERPs

## Requisitos

- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.23.0
- scipy >= 1.9.0
- matplotlib >= 3.6.0
- pulp >= 2.7.0 (otimização linear)

## Licença

MIT License
