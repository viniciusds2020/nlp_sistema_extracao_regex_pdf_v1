"""
Módulo de Geração de Dados Sintéticos
=====================================
Gera dados realistas de vendas para simulação do sistema de estoque.

Características dos dados gerados:
- 10 lojas com diferentes padrões de vendas
- 8 produtos com características distintas
- Sazonalidade semanal (fins de semana com mais vendas)
- Variação de 20-40% entre lojas
- Histórico de 30 dias
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple


# Configurações das lojas
LOJAS = [f"Loja_{str(i).zfill(2)}" for i in range(1, 11)]

# Configurações dos produtos com características específicas
PRODUTOS = {
    "PROD_001": {
        "nome": "Arroz 5kg",
        "demanda_base": 50,
        "preco_unitario": 25.90,
        "custo_manutencao": 0.50,  # Por unidade/dia
        "custo_stockout": 15.00,    # Por unidade não atendida
        "tempo_reposicao": 2,       # Dias
        "sazonalidade": "baixa"
    },
    "PROD_002": {
        "nome": "Feijão 1kg",
        "demanda_base": 80,
        "preco_unitario": 8.50,
        "custo_manutencao": 0.30,
        "custo_stockout": 10.00,
        "tempo_reposicao": 2,
        "sazonalidade": "baixa"
    },
    "PROD_003": {
        "nome": "Óleo de Soja 900ml",
        "demanda_base": 60,
        "preco_unitario": 7.90,
        "custo_manutencao": 0.25,
        "custo_stockout": 8.00,
        "tempo_reposicao": 3,
        "sazonalidade": "media"
    },
    "PROD_004": {
        "nome": "Açúcar 1kg",
        "demanda_base": 70,
        "preco_unitario": 4.50,
        "custo_manutencao": 0.20,
        "custo_stockout": 6.00,
        "tempo_reposicao": 2,
        "sazonalidade": "baixa"
    },
    "PROD_005": {
        "nome": "Refrigerante 2L",
        "demanda_base": 120,
        "preco_unitario": 9.90,
        "custo_manutencao": 0.40,
        "custo_stockout": 12.00,
        "tempo_reposicao": 1,
        "sazonalidade": "alta"  # Mais vendas no fim de semana
    },
    "PROD_006": {
        "nome": "Cerveja Pack 12",
        "demanda_base": 90,
        "preco_unitario": 45.00,
        "custo_manutencao": 0.80,
        "custo_stockout": 25.00,
        "tempo_reposicao": 1,
        "sazonalidade": "alta"
    },
    "PROD_007": {
        "nome": "Leite 1L",
        "demanda_base": 100,
        "preco_unitario": 5.50,
        "custo_manutencao": 0.60,  # Perecível
        "custo_stockout": 8.00,
        "tempo_reposicao": 1,
        "sazonalidade": "baixa"
    },
    "PROD_008": {
        "nome": "Detergente 500ml",
        "demanda_base": 40,
        "preco_unitario": 3.50,
        "custo_manutencao": 0.15,
        "custo_stockout": 5.00,
        "tempo_reposicao": 4,
        "sazonalidade": "media"
    }
}

# Perfis das lojas (multiplicador de demanda)
PERFIL_LOJAS = {
    "Loja_01": {"multiplicador": 1.4, "tipo": "grande"},      # Loja grande, alta demanda
    "Loja_02": {"multiplicador": 1.2, "tipo": "grande"},
    "Loja_03": {"multiplicador": 1.0, "tipo": "media"},
    "Loja_04": {"multiplicador": 0.9, "tipo": "media"},
    "Loja_05": {"multiplicador": 1.1, "tipo": "media"},
    "Loja_06": {"multiplicador": 0.8, "tipo": "pequena"},
    "Loja_07": {"multiplicador": 0.7, "tipo": "pequena"},
    "Loja_08": {"multiplicador": 1.3, "tipo": "grande"},
    "Loja_09": {"multiplicador": 0.85, "tipo": "media"},
    "Loja_10": {"multiplicador": 0.75, "tipo": "pequena"},
}


def calcular_fator_sazonalidade(dia_semana: int, tipo_sazonalidade: str) -> float:
    """
    Calcula o fator de sazonalidade baseado no dia da semana.

    Args:
        dia_semana: 0 = Segunda, 6 = Domingo
        tipo_sazonalidade: 'baixa', 'media' ou 'alta'

    Returns:
        Fator multiplicador para ajuste de demanda
    """
    # Fatores base por dia da semana
    fatores_base = {
        0: 0.85,   # Segunda - mais baixo
        1: 0.90,   # Terça
        2: 0.95,   # Quarta
        3: 1.00,   # Quinta
        4: 1.10,   # Sexta - começa a subir
        5: 1.25,   # Sábado - pico
        6: 1.15,   # Domingo
    }

    fator = fatores_base[dia_semana]

    # Amplifica ou reduz a sazonalidade conforme o tipo
    if tipo_sazonalidade == "alta":
        # Amplifica a variação
        fator = 1 + (fator - 1) * 1.5
    elif tipo_sazonalidade == "baixa":
        # Reduz a variação
        fator = 1 + (fator - 1) * 0.3

    return fator


def gerar_historico_vendas(
    dias: int = 30,
    seed: int = 42
) -> pd.DataFrame:
    """
    Gera histórico de vendas diárias para todas as lojas e produtos.

    Args:
        dias: Número de dias de histórico a gerar
        seed: Seed para reprodutibilidade

    Returns:
        DataFrame com colunas: data, loja_id, produto_id, quantidade_vendida,
                               receita, dia_semana
    """
    np.random.seed(seed)

    # Data final é hoje, data inicial é 'dias' atrás
    data_final = datetime.now().date()
    datas = [data_final - timedelta(days=i) for i in range(dias-1, -1, -1)]

    registros = []

    for data in datas:
        dia_semana = data.weekday()

        for loja_id in LOJAS:
            perfil = PERFIL_LOJAS[loja_id]
            multiplicador_loja = perfil["multiplicador"]

            for produto_id, config in PRODUTOS.items():
                # Demanda base ajustada pelo perfil da loja
                demanda_base = config["demanda_base"] * multiplicador_loja

                # Aplicar sazonalidade semanal
                fator_sazonal = calcular_fator_sazonalidade(
                    dia_semana,
                    config["sazonalidade"]
                )
                demanda_ajustada = demanda_base * fator_sazonal

                # Adicionar variação aleatória (coeficiente de variação ~20-30%)
                desvio = demanda_ajustada * 0.25
                quantidade = max(0, int(np.random.normal(demanda_ajustada, desvio)))

                # Calcular receita
                receita = quantidade * config["preco_unitario"]

                registros.append({
                    "data": data,
                    "loja_id": loja_id,
                    "produto_id": produto_id,
                    "produto_nome": config["nome"],
                    "quantidade_vendida": quantidade,
                    "preco_unitario": config["preco_unitario"],
                    "receita": receita,
                    "dia_semana": dia_semana,
                    "nome_dia": ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"][dia_semana]
                })

    df = pd.DataFrame(registros)
    df["data"] = pd.to_datetime(df["data"])

    return df


def gerar_estoque_atual(
    historico_vendas: pd.DataFrame,
    seed: int = 42
) -> pd.DataFrame:
    """
    Gera níveis de estoque atual simulados para cada loja/produto.

    O estoque é calculado como uma proporção da demanda média recente,
    com alguma variação para criar cenários realistas.

    Args:
        historico_vendas: DataFrame com histórico de vendas
        seed: Seed para reprodutibilidade

    Returns:
        DataFrame com estoque atual por loja e produto
    """
    np.random.seed(seed)

    # Calcular demanda média dos últimos 7 dias
    ultimos_7_dias = historico_vendas["data"].max() - timedelta(days=7)
    vendas_recentes = historico_vendas[historico_vendas["data"] >= ultimos_7_dias]

    demanda_media = vendas_recentes.groupby(
        ["loja_id", "produto_id", "produto_nome"]
    )["quantidade_vendida"].mean().reset_index()

    demanda_media.columns = ["loja_id", "produto_id", "produto_nome", "demanda_media_diaria"]

    # Gerar estoque atual com variação
    registros_estoque = []

    for _, row in demanda_media.iterrows():
        produto_config = PRODUTOS[row["produto_id"]]

        # Estoque entre 3-15 dias de demanda (variação realista)
        dias_estoque = np.random.uniform(3, 15)
        estoque_atual = int(row["demanda_media_diaria"] * dias_estoque)

        # Alguns produtos podem estar em situação crítica
        if np.random.random() < 0.15:  # 15% de chance de estoque baixo
            estoque_atual = int(row["demanda_media_diaria"] * np.random.uniform(0.5, 2))

        registros_estoque.append({
            "loja_id": row["loja_id"],
            "produto_id": row["produto_id"],
            "produto_nome": row["produto_nome"],
            "estoque_atual": estoque_atual,
            "custo_manutencao": produto_config["custo_manutencao"],
            "custo_stockout": produto_config["custo_stockout"],
            "tempo_reposicao": produto_config["tempo_reposicao"],
            "demanda_media_diaria": round(row["demanda_media_diaria"], 2)
        })

    return pd.DataFrame(registros_estoque)


def get_produtos_info() -> pd.DataFrame:
    """
    Retorna DataFrame com informações dos produtos.

    Returns:
        DataFrame com detalhes de cada produto
    """
    registros = []
    for produto_id, config in PRODUTOS.items():
        registros.append({
            "produto_id": produto_id,
            "nome": config["nome"],
            "preco_unitario": config["preco_unitario"],
            "custo_manutencao": config["custo_manutencao"],
            "custo_stockout": config["custo_stockout"],
            "tempo_reposicao": config["tempo_reposicao"],
            "demanda_base": config["demanda_base"],
            "sazonalidade": config["sazonalidade"]
        })
    return pd.DataFrame(registros)


def get_lojas_info() -> pd.DataFrame:
    """
    Retorna DataFrame com informações das lojas.

    Returns:
        DataFrame com detalhes de cada loja
    """
    registros = []
    for loja_id, perfil in PERFIL_LOJAS.items():
        registros.append({
            "loja_id": loja_id,
            "tipo": perfil["tipo"],
            "multiplicador_demanda": perfil["multiplicador"]
        })
    return pd.DataFrame(registros)


if __name__ == "__main__":
    # Teste de geração de dados
    print("Gerando dados de exemplo...")

    historico = gerar_historico_vendas(dias=30)
    print(f"\nHistórico de vendas: {len(historico)} registros")
    print(historico.head(10))

    estoque = gerar_estoque_atual(historico)
    print(f"\nEstoque atual: {len(estoque)} registros")
    print(estoque.head(10))

    produtos = get_produtos_info()
    print(f"\nProdutos: {len(produtos)} registros")
    print(produtos)

    lojas = get_lojas_info()
    print(f"\nLojas: {len(lojas)} registros")
    print(lojas)
