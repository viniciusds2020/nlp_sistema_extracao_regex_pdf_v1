"""
Módulo de Otimização de Estoque
===============================
Gera recomendações de reposição e otimiza níveis de estoque.

Funcionalidades:
- Geração de recomendações de reposição priorizadas
- Cálculo de custos (manutenção vs stockout)
- Alertas de estoque crítico
- Simulação de cenários what-if
- Otimização de quantidade de pedido (EOQ)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Prioridade(Enum):
    """Níveis de prioridade para reposição."""
    URGENTE = 1
    ALTA = 2
    MEDIA = 3
    BAIXA = 4
    MONITORAR = 5


@dataclass
class RecomendacaoReposicao:
    """Estrutura para recomendação de reposição."""
    loja_id: str
    produto_id: str
    produto_nome: str
    quantidade_recomendada: int
    prioridade: Prioridade
    dias_ate_stockout: float
    custo_estimado_stockout: float
    custo_manutencao_atual: float
    justificativa: str


def calcular_eoq(
    demanda_anual: float,
    custo_pedido: float,
    custo_manutencao_unitario: float
) -> float:
    """
    Calcula a Quantidade Econômica de Pedido (EOQ).

    Fórmula: EOQ = √(2 * D * S / H)
    Onde:
        - D = demanda anual
        - S = custo por pedido
        - H = custo de manutenção por unidade/ano

    Args:
        demanda_anual: Demanda anual estimada
        custo_pedido: Custo fixo por pedido
        custo_manutencao_unitario: Custo de manutenção por unidade/ano

    Returns:
        Quantidade econômica de pedido
    """
    if custo_manutencao_unitario <= 0 or demanda_anual <= 0:
        return 0

    eoq = np.sqrt(
        (2 * demanda_anual * custo_pedido) / custo_manutencao_unitario
    )
    return round(eoq, 0)


def calcular_custo_total_estoque(
    nivel_estoque: float,
    demanda_diaria: float,
    custo_manutencao: float,
    custo_stockout: float,
    probabilidade_stockout: float,
    horizonte_dias: int = 30
) -> Dict[str, float]:
    """
    Calcula o custo total de estoque (manutenção + risco de falta).

    Args:
        nivel_estoque: Nível médio de estoque
        demanda_diaria: Demanda média diária
        custo_manutencao: Custo de manutenção por unidade/dia
        custo_stockout: Custo por unidade em falta
        probabilidade_stockout: Probabilidade de stockout
        horizonte_dias: Horizonte de análise em dias

    Returns:
        Dict com custos detalhados
    """
    # Custo de manutenção (estoque médio * custo * dias)
    custo_manutencao_total = nivel_estoque * custo_manutencao * horizonte_dias

    # Custo esperado de stockout
    demanda_periodo = demanda_diaria * horizonte_dias
    unidades_falta_esperadas = demanda_periodo * probabilidade_stockout
    custo_stockout_total = unidades_falta_esperadas * custo_stockout

    return {
        "custo_manutencao": round(custo_manutencao_total, 2),
        "custo_stockout_esperado": round(custo_stockout_total, 2),
        "custo_total": round(custo_manutencao_total + custo_stockout_total, 2),
        "unidades_falta_esperadas": round(unidades_falta_esperadas, 1)
    }


def gerar_recomendacoes_reposicao(
    metricas_estoque: pd.DataFrame,
    custo_pedido: float = 50.0
) -> List[RecomendacaoReposicao]:
    """
    Gera lista de recomendações de reposição priorizadas.

    Args:
        metricas_estoque: DataFrame com métricas de estoque
        custo_pedido: Custo fixo por pedido

    Returns:
        Lista de RecomendacaoReposicao ordenada por prioridade
    """
    recomendacoes = []

    for _, row in metricas_estoque.iterrows():
        # Calcular dias até stockout
        if row["demanda_media"] > 0:
            dias_ate_stockout = row["estoque_atual"] / row["demanda_media"]
        else:
            dias_ate_stockout = float("inf")

        # Determinar se precisa reposição
        precisa_reposicao = (
            row["estoque_atual"] <= row["ponto_reorder"] or
            row["status"] in ["CRÍTICO", "ALERTA"]
        )

        if precisa_reposicao:
            # Calcular quantidade recomendada
            # Objetivo: levar estoque para ponto de reorder + EOQ
            demanda_anual = row["demanda_media"] * 365
            custo_manutencao_anual = row["custo_manutencao"] * 365

            eoq = calcular_eoq(
                demanda_anual,
                custo_pedido,
                custo_manutencao_anual
            )

            # Quantidade mínima: cobrir lead time + estoque de segurança
            quantidade_minima = max(
                row["ponto_reorder"] - row["estoque_atual"],
                row["demanda_media"] * row["tempo_reposicao"]
            )

            # Quantidade recomendada: máximo entre EOQ e mínimo necessário
            quantidade_recomendada = max(int(eoq), int(quantidade_minima))

            # Definir prioridade
            if row["status"] == "CRÍTICO" or dias_ate_stockout <= 1:
                prioridade = Prioridade.URGENTE
                justificativa = f"Estoque crítico! Apenas {dias_ate_stockout:.1f} dias de cobertura."
            elif row["status"] == "ALERTA" or dias_ate_stockout <= row["tempo_reposicao"]:
                prioridade = Prioridade.ALTA
                justificativa = f"Estoque abaixo do ponto de reorder. Reposição necessária em {dias_ate_stockout:.1f} dias."
            elif dias_ate_stockout <= row["tempo_reposicao"] * 2:
                prioridade = Prioridade.MEDIA
                justificativa = f"Estoque adequado por {dias_ate_stockout:.1f} dias. Planejar reposição."
            else:
                prioridade = Prioridade.BAIXA
                justificativa = f"Estoque confortável. Monitorar para próximo ciclo."

            # Calcular custos
            custo_stockout_estimado = (
                max(0, row["demanda_media"] * row["tempo_reposicao"] - row["estoque_atual"]) *
                row["custo_stockout"]
            )
            custo_manutencao_atual = row["estoque_atual"] * row["custo_manutencao"]

            recomendacoes.append(RecomendacaoReposicao(
                loja_id=row["loja_id"],
                produto_id=row["produto_id"],
                produto_nome=row["produto_nome"],
                quantidade_recomendada=quantidade_recomendada,
                prioridade=prioridade,
                dias_ate_stockout=round(dias_ate_stockout, 1),
                custo_estimado_stockout=round(custo_stockout_estimado, 2),
                custo_manutencao_atual=round(custo_manutencao_atual, 2),
                justificativa=justificativa
            ))

    # Ordenar por prioridade
    recomendacoes.sort(key=lambda x: (x.prioridade.value, -x.custo_estimado_stockout))

    return recomendacoes


def gerar_alertas_estoque(
    metricas_estoque: pd.DataFrame
) -> pd.DataFrame:
    """
    Gera alertas para itens com estoque crítico ou em alerta.

    Args:
        metricas_estoque: DataFrame com métricas de estoque

    Returns:
        DataFrame com alertas filtrados e ordenados
    """
    alertas = metricas_estoque[
        metricas_estoque["status"].isin(["CRÍTICO", "ALERTA"])
    ].copy()

    # Ordenar por criticidade
    ordem_status = {"CRÍTICO": 0, "ALERTA": 1}
    alertas["ordem"] = alertas["status"].map(ordem_status)
    alertas = alertas.sort_values(["ordem", "dias_estoque"])
    alertas = alertas.drop(columns=["ordem"])

    # Adicionar mensagem de alerta
    def gerar_mensagem(row):
        if row["status"] == "CRÍTICO":
            return f"⚠️ CRÍTICO: Apenas {row['dias_estoque']} dias de estoque!"
        else:
            return f"⚡ ALERTA: Estoque abaixo do ponto de reorder ({row['estoque_atual']} < {row['ponto_reorder']})"

    alertas["mensagem"] = alertas.apply(gerar_mensagem, axis=1)

    return alertas


def simular_cenario_what_if(
    metricas_estoque: pd.DataFrame,
    fator_demanda: float = 1.0,
    dias_simulacao: int = 30
) -> pd.DataFrame:
    """
    Simula cenário hipotético com alteração na demanda.

    Args:
        metricas_estoque: DataFrame com métricas atuais
        fator_demanda: Multiplicador de demanda (1.2 = +20%, 0.8 = -20%)
        dias_simulacao: Horizonte da simulação em dias

    Returns:
        DataFrame com resultados da simulação
    """
    simulacao = metricas_estoque.copy()

    # Ajustar demanda
    simulacao["demanda_simulada"] = simulacao["demanda_media"] * fator_demanda

    # Recalcular dias de estoque
    simulacao["dias_estoque_simulado"] = (
        simulacao["estoque_atual"] / simulacao["demanda_simulada"]
    ).round(1)

    # Calcular novo ponto de reorder
    from scipy import stats
    z = stats.norm.ppf(simulacao["nivel_servico"].iloc[0])

    simulacao["estoque_seguranca_simulado"] = (
        z * simulacao["demanda_std"] * fator_demanda *
        np.sqrt(simulacao["tempo_reposicao"])
    ).round(0).astype(int)

    simulacao["ponto_reorder_simulado"] = (
        simulacao["demanda_simulada"] * simulacao["tempo_reposicao"] +
        simulacao["estoque_seguranca_simulado"]
    ).round(0).astype(int)

    # Novo status
    def calcular_status_simulado(row):
        if row["estoque_atual"] <= row["estoque_seguranca_simulado"]:
            return "CRÍTICO"
        elif row["estoque_atual"] <= row["ponto_reorder_simulado"]:
            return "ALERTA"
        elif row["dias_estoque_simulado"] > 30:
            return "EXCESSO"
        else:
            return "ADEQUADO"

    simulacao["status_simulado"] = simulacao.apply(calcular_status_simulado, axis=1)

    # Calcular impacto nos custos
    simulacao["custo_manutencao_diario"] = (
        simulacao["estoque_atual"] * simulacao["custo_manutencao"]
    )

    # Probabilidade de stockout simplificada
    simulacao["prob_stockout"] = np.where(
        simulacao["dias_estoque_simulado"] < simulacao["tempo_reposicao"],
        (simulacao["tempo_reposicao"] - simulacao["dias_estoque_simulado"]) /
        simulacao["tempo_reposicao"],
        0
    ).clip(0, 1)

    simulacao["custo_stockout_esperado"] = (
        simulacao["prob_stockout"] *
        simulacao["demanda_simulada"] *
        simulacao["tempo_reposicao"] *
        simulacao["custo_stockout"]
    ).round(2)

    # Custo total esperado para o período
    simulacao["custo_total_periodo"] = (
        simulacao["custo_manutencao_diario"] * dias_simulacao +
        simulacao["custo_stockout_esperado"]
    ).round(2)

    return simulacao


def calcular_relatorio_custos(
    metricas_estoque: pd.DataFrame,
    horizonte_dias: int = 30
) -> Dict:
    """
    Gera relatório detalhado de custos de estoque.

    Args:
        metricas_estoque: DataFrame com métricas de estoque
        horizonte_dias: Horizonte de análise em dias

    Returns:
        Dict com custos agregados e por categoria
    """
    df = metricas_estoque.copy()

    # Custo de manutenção
    df["custo_manutencao_periodo"] = (
        df["estoque_atual"] * df["custo_manutencao"] * horizonte_dias
    )

    # Custo de stockout potencial (itens em situação crítica)
    df["risco_stockout"] = np.where(
        df["dias_estoque"] < df["tempo_reposicao"],
        (df["tempo_reposicao"] - df["dias_estoque"]) * df["demanda_media"] * df["custo_stockout"],
        0
    )

    # Agregações
    custo_total_manutencao = df["custo_manutencao_periodo"].sum()
    custo_total_risco = df["risco_stockout"].sum()

    # Por loja
    custos_por_loja = df.groupby("loja_id").agg({
        "custo_manutencao_periodo": "sum",
        "risco_stockout": "sum",
        "estoque_atual": "sum"
    }).reset_index()
    custos_por_loja["custo_total"] = (
        custos_por_loja["custo_manutencao_periodo"] +
        custos_por_loja["risco_stockout"]
    )

    # Por produto
    custos_por_produto = df.groupby(["produto_id", "produto_nome"]).agg({
        "custo_manutencao_periodo": "sum",
        "risco_stockout": "sum",
        "estoque_atual": "sum"
    }).reset_index()
    custos_por_produto["custo_total"] = (
        custos_por_produto["custo_manutencao_periodo"] +
        custos_por_produto["risco_stockout"]
    )

    # Por status
    custos_por_status = df.groupby("status").agg({
        "loja_id": "count",
        "custo_manutencao_periodo": "sum",
        "risco_stockout": "sum"
    }).reset_index()
    custos_por_status.columns = ["status", "quantidade_itens", "custo_manutencao", "risco_stockout"]

    return {
        "resumo": {
            "horizonte_dias": horizonte_dias,
            "total_itens": len(df),
            "custo_manutencao_total": round(custo_total_manutencao, 2),
            "risco_stockout_total": round(custo_total_risco, 2),
            "custo_total_estimado": round(custo_total_manutencao + custo_total_risco, 2),
            "itens_criticos": len(df[df["status"] == "CRÍTICO"]),
            "itens_alerta": len(df[df["status"] == "ALERTA"]),
            "itens_excesso": len(df[df["status"] == "EXCESSO"])
        },
        "por_loja": custos_por_loja.to_dict(orient="records"),
        "por_produto": custos_por_produto.to_dict(orient="records"),
        "por_status": custos_por_status.to_dict(orient="records")
    }


def recomendacoes_para_dataframe(
    recomendacoes: List[RecomendacaoReposicao]
) -> pd.DataFrame:
    """
    Converte lista de recomendações para DataFrame.

    Args:
        recomendacoes: Lista de RecomendacaoReposicao

    Returns:
        DataFrame com recomendações
    """
    if not recomendacoes:
        return pd.DataFrame()

    dados = []
    for rec in recomendacoes:
        dados.append({
            "loja_id": rec.loja_id,
            "produto_id": rec.produto_id,
            "produto_nome": rec.produto_nome,
            "quantidade_recomendada": rec.quantidade_recomendada,
            "prioridade": rec.prioridade.name,
            "dias_ate_stockout": rec.dias_ate_stockout,
            "custo_estimado_stockout": rec.custo_estimado_stockout,
            "custo_manutencao_atual": rec.custo_manutencao_atual,
            "justificativa": rec.justificativa
        })

    return pd.DataFrame(dados)


if __name__ == "__main__":
    # Teste das funções
    from data_generator import gerar_historico_vendas, gerar_estoque_atual
    from statistical_analysis import calcular_estatisticas_demanda, calcular_metricas_estoque

    print("Gerando dados de teste...")
    historico = gerar_historico_vendas(dias=30)
    estoque = gerar_estoque_atual(historico)
    stats_demanda = calcular_estatisticas_demanda(historico)
    metricas = calcular_metricas_estoque(estoque, stats_demanda)

    print("\n=== Recomendações de Reposição ===")
    recomendacoes = gerar_recomendacoes_reposicao(metricas)
    print(f"Total de recomendações: {len(recomendacoes)}")
    for rec in recomendacoes[:5]:
        print(f"\n{rec.loja_id} - {rec.produto_nome}")
        print(f"  Prioridade: {rec.prioridade.name}")
        print(f"  Quantidade: {rec.quantidade_recomendada}")
        print(f"  {rec.justificativa}")

    print("\n=== Alertas de Estoque ===")
    alertas = gerar_alertas_estoque(metricas)
    print(f"Total de alertas: {len(alertas)}")
    print(alertas[["loja_id", "produto_nome", "status", "dias_estoque", "mensagem"]].head(10))

    print("\n=== Simulação What-If (+30% demanda) ===")
    simulacao = simular_cenario_what_if(metricas, fator_demanda=1.3)
    print("Status atual vs simulado:")
    comparacao = simulacao.groupby(["status", "status_simulado"]).size().unstack(fill_value=0)
    print(comparacao)

    print("\n=== Relatório de Custos ===")
    relatorio = calcular_relatorio_custos(metricas)
    print(f"Custo manutenção (30 dias): R$ {relatorio['resumo']['custo_manutencao_total']:,.2f}")
    print(f"Risco stockout: R$ {relatorio['resumo']['risco_stockout_total']:,.2f}")
    print(f"Custo total estimado: R$ {relatorio['resumo']['custo_total_estimado']:,.2f}")
