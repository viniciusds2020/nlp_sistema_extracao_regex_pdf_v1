"""
Módulo de Visualização e Relatórios
===================================
Gera gráficos, dashboards e relatórios para análise de estoque.

Funcionalidades:
- Dashboard de status de estoque por loja
- Gráficos de tendência de demanda
- Relatórios formatados em texto
- Exportação para arquivos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional
from datetime import datetime
import os


# Configuração de estilo dos gráficos
plt.style.use('seaborn-v0_8-whitegrid')
CORES_STATUS = {
    "CRÍTICO": "#dc3545",
    "ALERTA": "#ffc107",
    "ADEQUADO": "#28a745",
    "EXCESSO": "#17a2b8"
}

CORES_PRIORIDADE = {
    "URGENTE": "#dc3545",
    "ALTA": "#fd7e14",
    "MEDIA": "#ffc107",
    "BAIXA": "#28a745",
    "MONITORAR": "#6c757d"
}


def criar_dashboard_estoque(
    metricas_estoque: pd.DataFrame,
    salvar_path: Optional[str] = None
) -> plt.Figure:
    """
    Cria dashboard visual com status de estoque por loja.

    Args:
        metricas_estoque: DataFrame com métricas de estoque
        salvar_path: Caminho para salvar a figura (opcional)

    Returns:
        Figura matplotlib
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Dashboard de Status de Estoque", fontsize=16, fontweight='bold')

    # 1. Status por loja (gráfico de barras empilhadas)
    ax1 = axes[0, 0]
    status_por_loja = metricas_estoque.groupby(
        ["loja_id", "status"]
    ).size().unstack(fill_value=0)

    # Reordenar colunas
    colunas_ordem = ["CRÍTICO", "ALERTA", "ADEQUADO", "EXCESSO"]
    colunas_presentes = [c for c in colunas_ordem if c in status_por_loja.columns]
    status_por_loja = status_por_loja[colunas_presentes]

    cores = [CORES_STATUS[c] for c in colunas_presentes]
    status_por_loja.plot(kind="bar", stacked=True, ax=ax1, color=cores, edgecolor='white')
    ax1.set_title("Status de Estoque por Loja", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Loja")
    ax1.set_ylabel("Quantidade de Produtos")
    ax1.legend(title="Status", bbox_to_anchor=(1.02, 1), loc='upper left')
    ax1.tick_params(axis='x', rotation=45)

    # 2. Distribuição de dias de estoque
    ax2 = axes[0, 1]
    df_plot = metricas_estoque.copy()
    df_plot["cor"] = df_plot["status"].map(CORES_STATUS)

    for status in colunas_presentes:
        dados = df_plot[df_plot["status"] == status]["dias_estoque"]
        ax2.hist(dados, bins=15, alpha=0.7, label=status, color=CORES_STATUS[status])

    ax2.set_title("Distribuição de Dias de Estoque", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Dias de Estoque")
    ax2.set_ylabel("Frequência")
    ax2.legend(title="Status")
    ax2.axvline(x=7, color='red', linestyle='--', label='Limite crítico (7 dias)')

    # 3. Top 10 produtos com menor cobertura
    ax3 = axes[1, 0]
    top_criticos = metricas_estoque.nsmallest(10, "dias_estoque")[
        ["loja_id", "produto_nome", "dias_estoque", "status"]
    ]
    cores_barras = [CORES_STATUS[s] for s in top_criticos["status"]]
    labels = [f"{row['loja_id']}\n{row['produto_nome']}" for _, row in top_criticos.iterrows()]

    bars = ax3.barh(range(len(top_criticos)), top_criticos["dias_estoque"], color=cores_barras)
    ax3.set_yticks(range(len(top_criticos)))
    ax3.set_yticklabels(labels, fontsize=8)
    ax3.set_title("Top 10 Itens com Menor Cobertura", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Dias de Estoque")
    ax3.invert_yaxis()

    # Adicionar valores nas barras
    for bar, valor in zip(bars, top_criticos["dias_estoque"]):
        ax3.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{valor:.1f}', va='center', fontsize=8)

    # 4. Resumo geral (texto/métricas)
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Calcular métricas resumo
    total_itens = len(metricas_estoque)
    criticos = len(metricas_estoque[metricas_estoque["status"] == "CRÍTICO"])
    alertas = len(metricas_estoque[metricas_estoque["status"] == "ALERTA"])
    adequados = len(metricas_estoque[metricas_estoque["status"] == "ADEQUADO"])
    excesso = len(metricas_estoque[metricas_estoque["status"] == "EXCESSO"])

    media_dias = metricas_estoque["dias_estoque"].mean()
    estoque_total = metricas_estoque["estoque_atual"].sum()

    resumo_texto = f"""
    RESUMO GERAL DO ESTOQUE
    {'='*40}

    Total de Itens Monitorados: {total_itens}

    Status:
    {'─'*30}
    🔴 Críticos:    {criticos:>5} ({criticos/total_itens*100:.1f}%)
    🟡 Em Alerta:   {alertas:>5} ({alertas/total_itens*100:.1f}%)
    🟢 Adequados:   {adequados:>5} ({adequados/total_itens*100:.1f}%)
    🔵 Em Excesso:  {excesso:>5} ({excesso/total_itens*100:.1f}%)

    Métricas:
    {'─'*30}
    Média de Dias de Estoque: {media_dias:.1f} dias
    Estoque Total (unidades): {estoque_total:,}

    Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}
    """

    ax4.text(0.1, 0.5, resumo_texto, transform=ax4.transAxes,
            fontsize=11, verticalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if salvar_path:
        plt.savefig(salvar_path, dpi=150, bbox_inches='tight')
        print(f"Dashboard salvo em: {salvar_path}")

    return fig


def plotar_tendencia_demanda(
    historico_vendas: pd.DataFrame,
    produto_id: str,
    loja_id: Optional[str] = None,
    salvar_path: Optional[str] = None
) -> plt.Figure:
    """
    Plota gráfico de tendência de demanda para um produto.

    Args:
        historico_vendas: DataFrame com histórico de vendas
        produto_id: ID do produto
        loja_id: ID da loja (opcional)
        salvar_path: Caminho para salvar a figura

    Returns:
        Figura matplotlib
    """
    df = historico_vendas[historico_vendas["produto_id"] == produto_id].copy()
    titulo_loja = "Todas as Lojas"

    if loja_id:
        df = df[df["loja_id"] == loja_id]
        titulo_loja = loja_id

    # Agregar por data
    df_agg = df.groupby("data").agg({
        "quantidade_vendida": ["sum", "mean", "std"]
    }).reset_index()
    df_agg.columns = ["data", "total", "media", "std"]
    df_agg = df_agg.sort_values("data")

    # Obter nome do produto
    nome_produto = df["produto_nome"].iloc[0] if len(df) > 0 else produto_id

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Gráfico 1: Vendas diárias com média móvel
    ax1 = axes[0]
    ax1.plot(df_agg["data"], df_agg["total"], marker='o', markersize=4,
             label="Vendas Diárias", alpha=0.7, linewidth=1)

    # Média móvel de 7 dias
    df_agg["media_movel_7d"] = df_agg["total"].rolling(window=7, min_periods=1).mean()
    ax1.plot(df_agg["data"], df_agg["media_movel_7d"],
             color='red', linewidth=2, label="Média Móvel (7 dias)")

    # Linha de tendência
    z = np.polyfit(range(len(df_agg)), df_agg["total"], 1)
    p = np.poly1d(z)
    ax1.plot(df_agg["data"], p(range(len(df_agg))),
             color='green', linestyle='--', linewidth=2, label="Tendência Linear")

    ax1.set_title(f"Tendência de Demanda: {nome_produto} ({titulo_loja})",
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel("Data")
    ax1.set_ylabel("Quantidade Vendida")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Gráfico 2: Demanda por dia da semana
    ax2 = axes[1]
    dias_semana = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]

    demanda_dia = df.groupby("dia_semana")["quantidade_vendida"].agg(["mean", "std"]).reset_index()
    demanda_dia = demanda_dia.sort_values("dia_semana")

    cores_dias = plt.cm.Blues(np.linspace(0.3, 0.9, 7))
    bars = ax2.bar(dias_semana, demanda_dia["mean"], yerr=demanda_dia["std"],
                   color=cores_dias, edgecolor='navy', capsize=3, alpha=0.8)

    ax2.set_title(f"Padrão Semanal: {nome_produto}", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Dia da Semana")
    ax2.set_ylabel("Demanda Média")

    # Destacar fim de semana
    for i, bar in enumerate(bars):
        if i >= 5:  # Sábado e Domingo
            bar.set_color('#ff7f0e')

    # Adicionar valores nas barras
    for bar, valor in zip(bars, demanda_dia["mean"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + demanda_dia["std"].max()*0.1,
                f'{valor:.0f}', ha='center', fontsize=9)

    plt.tight_layout()

    if salvar_path:
        plt.savefig(salvar_path, dpi=150, bbox_inches='tight')
        print(f"Gráfico salvo em: {salvar_path}")

    return fig


def plotar_comparativo_lojas(
    metricas_estoque: pd.DataFrame,
    salvar_path: Optional[str] = None
) -> plt.Figure:
    """
    Cria gráfico comparativo de estoque entre lojas.

    Args:
        metricas_estoque: DataFrame com métricas de estoque
        salvar_path: Caminho para salvar a figura

    Returns:
        Figura matplotlib
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Estoque total por loja
    ax1 = axes[0]
    estoque_loja = metricas_estoque.groupby("loja_id").agg({
        "estoque_atual": "sum",
        "dias_estoque": "mean"
    }).reset_index()
    estoque_loja = estoque_loja.sort_values("estoque_atual", ascending=True)

    cores = plt.cm.viridis(np.linspace(0.2, 0.8, len(estoque_loja)))
    bars = ax1.barh(estoque_loja["loja_id"], estoque_loja["estoque_atual"], color=cores)

    ax1.set_title("Estoque Total por Loja (unidades)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Unidades em Estoque")

    for bar, valor in zip(bars, estoque_loja["estoque_atual"]):
        ax1.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                f'{int(valor):,}', va='center', fontsize=9)

    # 2. Média de dias de estoque por loja
    ax2 = axes[1]
    estoque_loja = estoque_loja.sort_values("dias_estoque", ascending=True)

    cores_dias = []
    for dias in estoque_loja["dias_estoque"]:
        if dias < 5:
            cores_dias.append(CORES_STATUS["CRÍTICO"])
        elif dias < 10:
            cores_dias.append(CORES_STATUS["ALERTA"])
        elif dias > 20:
            cores_dias.append(CORES_STATUS["EXCESSO"])
        else:
            cores_dias.append(CORES_STATUS["ADEQUADO"])

    bars = ax2.barh(estoque_loja["loja_id"], estoque_loja["dias_estoque"], color=cores_dias)

    ax2.set_title("Média de Dias de Estoque por Loja", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Dias de Estoque")
    ax2.axvline(x=7, color='red', linestyle='--', alpha=0.7, label='Limite crítico')
    ax2.axvline(x=14, color='orange', linestyle='--', alpha=0.7, label='Limite alerta')

    for bar, valor in zip(bars, estoque_loja["dias_estoque"]):
        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{valor:.1f}', va='center', fontsize=9)

    ax2.legend()

    plt.tight_layout()

    if salvar_path:
        plt.savefig(salvar_path, dpi=150, bbox_inches='tight')
        print(f"Gráfico salvo em: {salvar_path}")

    return fig


def gerar_relatorio_texto(
    metricas_estoque: pd.DataFrame,
    recomendacoes_df: pd.DataFrame,
    relatorio_custos: Dict,
    salvar_path: Optional[str] = None
) -> str:
    """
    Gera relatório em formato texto.

    Args:
        metricas_estoque: DataFrame com métricas de estoque
        recomendacoes_df: DataFrame com recomendações
        relatorio_custos: Dict com informações de custos
        salvar_path: Caminho para salvar o arquivo

    Returns:
        String com relatório formatado
    """
    linhas = []
    linhas.append("=" * 80)
    linhas.append("           RELATÓRIO DE OTIMIZAÇÃO DE ESTOQUE")
    linhas.append("=" * 80)
    linhas.append(f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    linhas.append("")

    # Seção 1: Resumo Geral
    linhas.append("-" * 80)
    linhas.append("1. RESUMO GERAL")
    linhas.append("-" * 80)
    resumo = relatorio_custos["resumo"]
    linhas.append(f"   Total de itens monitorados: {resumo['total_itens']}")
    linhas.append(f"   Horizonte de análise: {resumo['horizonte_dias']} dias")
    linhas.append("")
    linhas.append("   STATUS DOS ITENS:")
    linhas.append(f"   - Críticos:  {resumo['itens_criticos']:>4} itens")
    linhas.append(f"   - Em Alerta: {resumo['itens_alerta']:>4} itens")
    linhas.append(f"   - Em Excesso:{resumo['itens_excesso']:>4} itens")
    linhas.append("")

    # Seção 2: Análise de Custos
    linhas.append("-" * 80)
    linhas.append("2. ANÁLISE DE CUSTOS")
    linhas.append("-" * 80)
    linhas.append(f"   Custo de manutenção (período): R$ {resumo['custo_manutencao_total']:>12,.2f}")
    linhas.append(f"   Risco de stockout:             R$ {resumo['risco_stockout_total']:>12,.2f}")
    linhas.append(f"   " + "-" * 40)
    linhas.append(f"   CUSTO TOTAL ESTIMADO:          R$ {resumo['custo_total_estimado']:>12,.2f}")
    linhas.append("")

    # Custos por loja
    linhas.append("   CUSTOS POR LOJA:")
    linhas.append("   " + "-" * 60)
    linhas.append(f"   {'Loja':<12} {'Manutenção':>15} {'Risco':>15} {'Total':>15}")
    linhas.append("   " + "-" * 60)
    for loja in relatorio_custos["por_loja"]:
        linhas.append(
            f"   {loja['loja_id']:<12} "
            f"R$ {loja['custo_manutencao_periodo']:>11,.2f} "
            f"R$ {loja['risco_stockout']:>11,.2f} "
            f"R$ {loja['custo_total']:>11,.2f}"
        )
    linhas.append("")

    # Seção 3: Itens Críticos
    linhas.append("-" * 80)
    linhas.append("3. ITENS EM SITUAÇÃO CRÍTICA")
    linhas.append("-" * 80)

    criticos = metricas_estoque[metricas_estoque["status"] == "CRÍTICO"]
    if len(criticos) > 0:
        linhas.append(f"   {'Loja':<10} {'Produto':<25} {'Estoque':>10} {'Dias':>8}")
        linhas.append("   " + "-" * 60)
        for _, row in criticos.head(15).iterrows():
            linhas.append(
                f"   {row['loja_id']:<10} "
                f"{row['produto_nome'][:24]:<25} "
                f"{int(row['estoque_atual']):>10} "
                f"{row['dias_estoque']:>8.1f}"
            )
    else:
        linhas.append("   Nenhum item em situação crítica.")
    linhas.append("")

    # Seção 4: Recomendações Prioritárias
    linhas.append("-" * 80)
    linhas.append("4. RECOMENDAÇÕES DE REPOSIÇÃO (TOP 20)")
    linhas.append("-" * 80)

    if len(recomendacoes_df) > 0:
        linhas.append(f"   {'Prior.':<10} {'Loja':<10} {'Produto':<20} {'Qtd':>8} {'Dias':>6}")
        linhas.append("   " + "-" * 60)
        for _, row in recomendacoes_df.head(20).iterrows():
            linhas.append(
                f"   {row['prioridade']:<10} "
                f"{row['loja_id']:<10} "
                f"{row['produto_nome'][:19]:<20} "
                f"{int(row['quantidade_recomendada']):>8} "
                f"{row['dias_ate_stockout']:>6.1f}"
            )
    else:
        linhas.append("   Nenhuma recomendação de reposição necessária.")
    linhas.append("")

    # Seção 5: Itens em Excesso
    linhas.append("-" * 80)
    linhas.append("5. ITENS EM EXCESSO DE ESTOQUE")
    linhas.append("-" * 80)

    excesso = metricas_estoque[metricas_estoque["status"] == "EXCESSO"]
    if len(excesso) > 0:
        linhas.append("   Atenção: Os itens abaixo possuem mais de 30 dias de cobertura.")
        linhas.append(f"   {'Loja':<10} {'Produto':<25} {'Estoque':>10} {'Dias':>8}")
        linhas.append("   " + "-" * 60)
        for _, row in excesso.head(10).iterrows():
            linhas.append(
                f"   {row['loja_id']:<10} "
                f"{row['produto_nome'][:24]:<25} "
                f"{int(row['estoque_atual']):>10} "
                f"{row['dias_estoque']:>8.1f}"
            )
    else:
        linhas.append("   Nenhum item com excesso de estoque.")
    linhas.append("")

    linhas.append("=" * 80)
    linhas.append("                    FIM DO RELATÓRIO")
    linhas.append("=" * 80)

    relatorio = "\n".join(linhas)

    if salvar_path:
        with open(salvar_path, "w", encoding="utf-8") as f:
            f.write(relatorio)
        print(f"Relatório salvo em: {salvar_path}")

    return relatorio


def plotar_todos_graficos(
    historico_vendas: pd.DataFrame,
    metricas_estoque: pd.DataFrame,
    pasta_saida: str = "reports"
) -> List[str]:
    """
    Gera todos os gráficos e salva em uma pasta.

    Args:
        historico_vendas: DataFrame com histórico de vendas
        metricas_estoque: DataFrame com métricas de estoque
        pasta_saida: Pasta para salvar os arquivos

    Returns:
        Lista de caminhos dos arquivos gerados
    """
    os.makedirs(pasta_saida, exist_ok=True)
    arquivos = []

    # Dashboard principal
    path = os.path.join(pasta_saida, "dashboard_estoque.png")
    criar_dashboard_estoque(metricas_estoque, salvar_path=path)
    arquivos.append(path)
    plt.close()

    # Comparativo de lojas
    path = os.path.join(pasta_saida, "comparativo_lojas.png")
    plotar_comparativo_lojas(metricas_estoque, salvar_path=path)
    arquivos.append(path)
    plt.close()

    # Tendência dos principais produtos
    produtos = historico_vendas["produto_id"].unique()[:4]  # Top 4 produtos
    for produto_id in produtos:
        nome_arquivo = f"tendencia_{produto_id.lower()}.png"
        path = os.path.join(pasta_saida, nome_arquivo)
        plotar_tendencia_demanda(historico_vendas, produto_id, salvar_path=path)
        arquivos.append(path)
        plt.close()

    return arquivos


if __name__ == "__main__":
    # Teste das funções
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from data_generator import gerar_historico_vendas, gerar_estoque_atual
    from statistical_analysis import calcular_estatisticas_demanda, calcular_metricas_estoque
    from inventory_optimizer import (
        gerar_recomendacoes_reposicao,
        recomendacoes_para_dataframe,
        calcular_relatorio_custos
    )

    print("Gerando dados de teste...")
    historico = gerar_historico_vendas(dias=30)
    estoque = gerar_estoque_atual(historico)
    stats_demanda = calcular_estatisticas_demanda(historico)
    metricas = calcular_metricas_estoque(estoque, stats_demanda)

    print("\n=== Gerando Dashboard ===")
    criar_dashboard_estoque(metricas, salvar_path="reports/teste_dashboard.png")

    print("\n=== Gerando Gráfico de Tendência ===")
    plotar_tendencia_demanda(historico, "PROD_005", salvar_path="reports/teste_tendencia.png")

    print("\n=== Gerando Comparativo de Lojas ===")
    plotar_comparativo_lojas(metricas, salvar_path="reports/teste_comparativo.png")

    print("\n=== Gerando Relatório Texto ===")
    recomendacoes = gerar_recomendacoes_reposicao(metricas)
    recomendacoes_df = recomendacoes_para_dataframe(recomendacoes)
    relatorio_custos = calcular_relatorio_custos(metricas)

    relatorio = gerar_relatorio_texto(
        metricas, recomendacoes_df, relatorio_custos,
        salvar_path="reports/teste_relatorio.txt"
    )
    print(relatorio)

    plt.show()
