"""
Módulo de Análise Estatística
=============================
Realiza análises estatísticas sobre dados de vendas e demanda.

Funcionalidades:
- Cálculo de demanda média e desvio padrão
- Análise de padrões de sazonalidade
- Cálculo de estoque de segurança
- Identificação de tendências
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy import stats


def calcular_estatisticas_demanda(
    historico_vendas: pd.DataFrame,
    agrupar_por: list = ["loja_id", "produto_id"]
) -> pd.DataFrame:
    """
    Calcula estatísticas descritivas da demanda por grupo.

    Args:
        historico_vendas: DataFrame com histórico de vendas
        agrupar_por: Colunas para agrupamento

    Returns:
        DataFrame com estatísticas: média, desvio padrão, CV, min, max, mediana
    """
    stats_df = historico_vendas.groupby(agrupar_por).agg({
        "quantidade_vendida": [
            ("demanda_media", "mean"),
            ("demanda_std", "std"),
            ("demanda_min", "min"),
            ("demanda_max", "max"),
            ("demanda_mediana", "median"),
            ("total_vendas", "sum"),
            ("dias_analisados", "count")
        ]
    }).reset_index()

    # Achatar colunas multi-nível
    stats_df.columns = [
        col[0] if col[1] == "" else col[1]
        for col in stats_df.columns
    ]

    # Calcular coeficiente de variação (CV)
    stats_df["coef_variacao"] = (
        stats_df["demanda_std"] / stats_df["demanda_media"] * 100
    ).round(2)

    # Classificar variabilidade
    stats_df["variabilidade"] = pd.cut(
        stats_df["coef_variacao"],
        bins=[0, 20, 40, 100],
        labels=["Baixa", "Média", "Alta"]
    )

    return stats_df


def analisar_sazonalidade_semanal(
    historico_vendas: pd.DataFrame
) -> pd.DataFrame:
    """
    Analisa padrões de sazonalidade por dia da semana.

    Args:
        historico_vendas: DataFrame com histórico de vendas

    Returns:
        DataFrame com índices de sazonalidade por dia da semana e produto
    """
    # Média geral por produto
    media_geral = historico_vendas.groupby("produto_id")["quantidade_vendida"].mean()

    # Média por dia da semana e produto
    media_dia_semana = historico_vendas.groupby(
        ["produto_id", "dia_semana", "nome_dia"]
    )["quantidade_vendida"].mean().reset_index()

    # Calcular índice de sazonalidade (demanda do dia / média geral)
    media_dia_semana["indice_sazonalidade"] = media_dia_semana.apply(
        lambda row: row["quantidade_vendida"] / media_geral[row["produto_id"]],
        axis=1
    ).round(3)

    # Ordenar por dia da semana
    media_dia_semana = media_dia_semana.sort_values(["produto_id", "dia_semana"])

    return media_dia_semana


def calcular_estoque_seguranca(
    demanda_media: float,
    demanda_std: float,
    tempo_reposicao: int,
    nivel_servico: float = 0.95
) -> Tuple[float, float]:
    """
    Calcula o estoque de segurança usando modelo estatístico.

    Utiliza a fórmula: SS = Z * σ_LT
    Onde:
        - Z = fator de segurança (baseado no nível de serviço)
        - σ_LT = desvio padrão da demanda durante o lead time
        - σ_LT = σ_d * √(LT) para demanda com variância constante

    Args:
        demanda_media: Demanda média diária
        demanda_std: Desvio padrão da demanda diária
        tempo_reposicao: Tempo de reposição em dias (Lead Time)
        nivel_servico: Nível de serviço desejado (probabilidade de não ter stockout)

    Returns:
        Tuple com (estoque_seguranca, ponto_reorder)
    """
    # Fator Z para o nível de serviço
    z = stats.norm.ppf(nivel_servico)

    # Desvio padrão durante o lead time
    std_lead_time = demanda_std * np.sqrt(tempo_reposicao)

    # Estoque de segurança
    estoque_seguranca = z * std_lead_time

    # Ponto de reorder = demanda durante LT + estoque de segurança
    ponto_reorder = (demanda_media * tempo_reposicao) + estoque_seguranca

    return round(estoque_seguranca, 0), round(ponto_reorder, 0)


def calcular_metricas_estoque(
    estoque_atual: pd.DataFrame,
    stats_demanda: pd.DataFrame,
    nivel_servico: float = 0.95
) -> pd.DataFrame:
    """
    Calcula métricas completas de estoque incluindo estoque de segurança.

    Args:
        estoque_atual: DataFrame com níveis atuais de estoque
        stats_demanda: DataFrame com estatísticas de demanda
        nivel_servico: Nível de serviço desejado

    Returns:
        DataFrame com métricas completas de estoque
    """
    # Merge dos dados
    df = estoque_atual.merge(
        stats_demanda[["loja_id", "produto_id", "demanda_media", "demanda_std"]],
        on=["loja_id", "produto_id"],
        how="left"
    )

    # Calcular estoque de segurança e ponto de reorder para cada linha
    resultados = []
    for _, row in df.iterrows():
        ss, pr = calcular_estoque_seguranca(
            demanda_media=row["demanda_media"],
            demanda_std=row["demanda_std"] if pd.notna(row["demanda_std"]) else row["demanda_media"] * 0.25,
            tempo_reposicao=row["tempo_reposicao"],
            nivel_servico=nivel_servico
        )

        # Dias de estoque disponíveis
        dias_estoque = (
            row["estoque_atual"] / row["demanda_media"]
            if row["demanda_media"] > 0 else 0
        )

        # Status do estoque
        if row["estoque_atual"] <= ss:
            status = "CRÍTICO"
        elif row["estoque_atual"] <= pr:
            status = "ALERTA"
        elif dias_estoque > 30:
            status = "EXCESSO"
        else:
            status = "ADEQUADO"

        resultados.append({
            "loja_id": row["loja_id"],
            "produto_id": row["produto_id"],
            "produto_nome": row["produto_nome"],
            "estoque_atual": row["estoque_atual"],
            "demanda_media": round(row["demanda_media"], 2),
            "demanda_std": round(row["demanda_std"], 2) if pd.notna(row["demanda_std"]) else 0,
            "estoque_seguranca": int(ss),
            "ponto_reorder": int(pr),
            "dias_estoque": round(dias_estoque, 1),
            "tempo_reposicao": row["tempo_reposicao"],
            "custo_manutencao": row["custo_manutencao"],
            "custo_stockout": row["custo_stockout"],
            "status": status,
            "nivel_servico": nivel_servico
        })

    return pd.DataFrame(resultados)


def identificar_tendencia(
    historico_vendas: pd.DataFrame,
    produto_id: str,
    loja_id: Optional[str] = None
) -> Dict:
    """
    Identifica tendência de demanda usando regressão linear.

    Args:
        historico_vendas: DataFrame com histórico de vendas
        produto_id: ID do produto a analisar
        loja_id: ID da loja (opcional, se None analisa agregado)

    Returns:
        Dict com informações da tendência
    """
    # Filtrar dados
    df = historico_vendas[historico_vendas["produto_id"] == produto_id].copy()
    if loja_id:
        df = df[df["loja_id"] == loja_id]

    # Agregar por data
    df_agg = df.groupby("data")["quantidade_vendida"].sum().reset_index()
    df_agg = df_agg.sort_values("data")

    # Criar variável numérica para o tempo
    df_agg["dia_num"] = range(len(df_agg))

    # Regressão linear
    if len(df_agg) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df_agg["dia_num"],
            df_agg["quantidade_vendida"]
        )

        # Classificar tendência
        if p_value < 0.05:  # Significativo
            if slope > 0:
                tendencia = "CRESCENTE"
            else:
                tendencia = "DECRESCENTE"
        else:
            tendencia = "ESTÁVEL"

        # Variação percentual estimada
        media = df_agg["quantidade_vendida"].mean()
        variacao_percentual = (slope * len(df_agg)) / media * 100 if media > 0 else 0

        return {
            "produto_id": produto_id,
            "loja_id": loja_id or "TODAS",
            "tendencia": tendencia,
            "inclinacao": round(slope, 2),
            "r_quadrado": round(r_value ** 2, 3),
            "p_valor": round(p_value, 4),
            "variacao_percentual_periodo": round(variacao_percentual, 1),
            "significativo": p_value < 0.05
        }
    else:
        return {
            "produto_id": produto_id,
            "loja_id": loja_id or "TODAS",
            "tendencia": "DADOS_INSUFICIENTES",
            "inclinacao": 0,
            "r_quadrado": 0,
            "p_valor": 1,
            "variacao_percentual_periodo": 0,
            "significativo": False
        }


def analisar_correlacao_lojas(
    historico_vendas: pd.DataFrame,
    produto_id: str
) -> pd.DataFrame:
    """
    Analisa correlação de demanda entre lojas para um produto.

    Args:
        historico_vendas: DataFrame com histórico de vendas
        produto_id: ID do produto

    Returns:
        DataFrame com matriz de correlação entre lojas
    """
    df = historico_vendas[historico_vendas["produto_id"] == produto_id]

    # Pivot para ter lojas como colunas
    pivot = df.pivot_table(
        index="data",
        columns="loja_id",
        values="quantidade_vendida",
        aggfunc="sum"
    )

    # Calcular correlação
    correlacao = pivot.corr()

    return correlacao


def calcular_variabilidade_demanda(
    historico_vendas: pd.DataFrame
) -> pd.DataFrame:
    """
    Calcula métricas de variabilidade para classificar produtos.

    Usa classificação ABC baseada em volume e XYZ baseada em variabilidade.

    Args:
        historico_vendas: DataFrame com histórico de vendas

    Returns:
        DataFrame com classificação ABC-XYZ
    """
    # Estatísticas por produto
    stats_produto = historico_vendas.groupby("produto_id").agg({
        "quantidade_vendida": ["sum", "mean", "std"],
        "receita": "sum"
    }).reset_index()

    stats_produto.columns = ["produto_id", "total_vendas", "media_vendas",
                            "std_vendas", "receita_total"]

    # Classificação ABC (por receita)
    stats_produto = stats_produto.sort_values("receita_total", ascending=False)
    stats_produto["receita_acumulada"] = stats_produto["receita_total"].cumsum()
    stats_produto["percentual_acumulado"] = (
        stats_produto["receita_acumulada"] /
        stats_produto["receita_total"].sum() * 100
    )

    def classificar_abc(percentual):
        if percentual <= 80:
            return "A"
        elif percentual <= 95:
            return "B"
        else:
            return "C"

    stats_produto["classe_abc"] = stats_produto["percentual_acumulado"].apply(
        classificar_abc
    )

    # Classificação XYZ (por variabilidade)
    stats_produto["cv"] = (
        stats_produto["std_vendas"] / stats_produto["media_vendas"] * 100
    )

    def classificar_xyz(cv):
        if cv <= 20:
            return "X"  # Baixa variabilidade
        elif cv <= 50:
            return "Y"  # Média variabilidade
        else:
            return "Z"  # Alta variabilidade

    stats_produto["classe_xyz"] = stats_produto["cv"].apply(classificar_xyz)

    # Classificação combinada
    stats_produto["classe_abc_xyz"] = (
        stats_produto["classe_abc"] + stats_produto["classe_xyz"]
    )

    return stats_produto


if __name__ == "__main__":
    # Teste das funções
    from data_generator import gerar_historico_vendas, gerar_estoque_atual

    print("Gerando dados de teste...")
    historico = gerar_historico_vendas(dias=30)
    estoque = gerar_estoque_atual(historico)

    print("\n=== Estatísticas de Demanda ===")
    stats = calcular_estatisticas_demanda(historico)
    print(stats.head(10))

    print("\n=== Sazonalidade Semanal ===")
    sazonalidade = analisar_sazonalidade_semanal(historico)
    print(sazonalidade[sazonalidade["produto_id"] == "PROD_005"])

    print("\n=== Métricas de Estoque ===")
    metricas = calcular_metricas_estoque(estoque, stats)
    print(metricas.head(10))

    print("\n=== Análise de Tendência ===")
    tendencia = identificar_tendencia(historico, "PROD_001")
    print(tendencia)

    print("\n=== Classificação ABC-XYZ ===")
    abc_xyz = calcular_variabilidade_demanda(historico)
    print(abc_xyz)
