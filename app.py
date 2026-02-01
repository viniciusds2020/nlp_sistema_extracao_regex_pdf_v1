"""
Dashboard de Otimização de Estoque - Interface Streamlit
=========================================================
Interface web moderna estilo Tableau para gestão de estoque.

Executar com: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta

# Adiciona o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_generator import (
    gerar_historico_vendas,
    gerar_estoque_atual,
    LOJAS,
    PRODUTOS
)
from statistical_analysis import (
    calcular_estatisticas_demanda,
    calcular_metricas_estoque,
    analisar_sazonalidade_semanal
)
from inventory_optimizer import (
    gerar_recomendacoes_reposicao,
    gerar_alertas_estoque,
    simular_cenario_what_if,
    calcular_relatorio_custos,
    recomendacoes_para_dataframe
)

# Importações opcionais
try:
    from linear_optimization import OtimizadorLinear, PULP_AVAILABLE
    LINEAR_OPT_AVAILABLE = PULP_AVAILABLE
except ImportError:
    LINEAR_OPT_AVAILABLE = False

try:
    from forecasting import PrevisaoDemanda, previsoes_para_dataframe, STATSMODELS_AVAILABLE
    FORECASTING_AVAILABLE = STATSMODELS_AVAILABLE
except ImportError:
    FORECASTING_AVAILABLE = False

# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Inventory Optimizer",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS CUSTOMIZADO - ESTILO TABLEAU
# ============================================================================

st.markdown("""
<style>
    /* Importar fonte */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Reset e base */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Fundo principal */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
    }

    /* Sidebar estilo Tableau */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a365d 0%, #2c5282 100%);
        padding-top: 1rem;
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }

    [data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
    }

    /* Cards de métricas */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border-left: 4px solid;
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }

    .metric-card.blue { border-left-color: #3182ce; }
    .metric-card.green { border-left-color: #38a169; }
    .metric-card.orange { border-left-color: #dd6b20; }
    .metric-card.red { border-left-color: #e53e3e; }
    .metric-card.purple { border-left-color: #805ad5; }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a202c;
        line-height: 1;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        font-size: 0.875rem;
        color: #718096;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-delta {
        font-size: 0.875rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }

    .metric-delta.positive { color: #38a169; }
    .metric-delta.negative { color: #e53e3e; }

    /* Título principal */
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 0.5rem;
    }

    .sub-title {
        font-size: 1rem;
        color: #718096;
        margin-bottom: 2rem;
    }

    /* Cards de seção */
    .section-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }

    /* Tabelas estilizadas */
    .dataframe {
        font-size: 0.875rem;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }

    .status-critico { background: #fed7d7; color: #c53030; }
    .status-alerta { background: #feebc8; color: #c05621; }
    .status-adequado { background: #c6f6d5; color: #276749; }
    .status-excesso { background: #bee3f8; color: #2b6cb0; }

    /* Ajustes para tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        padding: 0.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: #3182ce !important;
        color: white !important;
    }

    /* Esconder elementos padrão do Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Alertas customizados */
    .alert-box {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .alert-critical {
        background: #fff5f5;
        border: 1px solid #feb2b2;
        color: #c53030;
    }

    .alert-warning {
        background: #fffaf0;
        border: 1px solid #fbd38d;
        color: #c05621;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

@st.cache_data(ttl=3600)
def carregar_dados(dias: int = 30, seed: int = 42):
    """Carrega e processa todos os dados necessários."""
    historico = gerar_historico_vendas(dias=dias, seed=seed)
    estoque = gerar_estoque_atual(historico, seed=seed)
    stats = calcular_estatisticas_demanda(historico)
    metricas = calcular_metricas_estoque(estoque, stats)
    return historico, estoque, stats, metricas


def criar_kpi_card(valor, label, delta=None, delta_color="positive", cor="blue"):
    """Cria um card de KPI estilizado."""
    delta_html = ""
    if delta:
        delta_class = "positive" if delta_color == "positive" else "negative"
        delta_html = f'<div class="metric-delta {delta_class}">{delta}</div>'

    return f"""
    <div class="metric-card {cor}">
        <div class="metric-value">{valor}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """


def formatar_moeda(valor):
    """Formata valor como moeda brasileira."""
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def get_cor_status(status):
    """Retorna cor baseada no status."""
    cores = {
        "CRÍTICO": "#e53e3e",
        "ALERTA": "#dd6b20",
        "ADEQUADO": "#38a169",
        "EXCESSO": "#3182ce"
    }
    return cores.get(status, "#718096")


# ============================================================================
# SIDEBAR - FILTROS E CONTROLES
# ============================================================================

with st.sidebar:
    # Logo e título
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">📦</div>
        <div style="font-size: 1.25rem; font-weight: 700; color: white;">Inventory</div>
        <div style="font-size: 1.25rem; font-weight: 300; color: #90cdf4;">Optimizer</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Configurações de dados
    st.markdown("### ⚙️ Configurações")

    dias_historico = st.slider(
        "Dias de histórico",
        min_value=14,
        max_value=90,
        value=30,
        step=7
    )

    seed = st.number_input(
        "Seed (reprodutibilidade)",
        min_value=1,
        max_value=9999,
        value=42
    )

    st.markdown("---")

    # Filtros
    st.markdown("### 🔍 Filtros")

    lojas_selecionadas = st.multiselect(
        "Lojas",
        options=LOJAS,
        default=LOJAS
    )

    produtos_selecionados = st.multiselect(
        "Produtos",
        options=list(PRODUTOS.keys()),
        default=list(PRODUTOS.keys()),
        format_func=lambda x: PRODUTOS[x]["nome"]
    )

    status_selecionados = st.multiselect(
        "Status",
        options=["CRÍTICO", "ALERTA", "ADEQUADO", "EXCESSO"],
        default=["CRÍTICO", "ALERTA", "ADEQUADO", "EXCESSO"]
    )

    st.markdown("---")

    # Info
    st.markdown("""
    <div style="font-size: 0.75rem; color: #a0aec0; text-align: center; padding: 1rem;">
        Versão 1.0<br>
        Atualizado em tempo real
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# CARREGAR DADOS
# ============================================================================

historico, estoque, stats, metricas = carregar_dados(dias_historico, seed)

# Aplicar filtros
metricas_filtradas = metricas[
    (metricas["loja_id"].isin(lojas_selecionadas)) &
    (metricas["produto_id"].isin(produtos_selecionados)) &
    (metricas["status"].isin(status_selecionados))
]

historico_filtrado = historico[
    (historico["loja_id"].isin(lojas_selecionadas)) &
    (historico["produto_id"].isin(produtos_selecionados))
]


# ============================================================================
# HEADER
# ============================================================================

st.markdown("""
<div style="margin-bottom: 2rem;">
    <div class="main-title">📊 Dashboard de Otimização de Estoque</div>
    <div class="sub-title">Monitoramento em tempo real • {} lojas • {} produtos • Última atualização: {}</div>
</div>
""".format(
    len(lojas_selecionadas),
    len(produtos_selecionados),
    datetime.now().strftime("%d/%m/%Y %H:%M")
), unsafe_allow_html=True)


# ============================================================================
# KPIs PRINCIPAIS
# ============================================================================

# Calcular métricas
total_itens = len(metricas_filtradas)
itens_criticos = len(metricas_filtradas[metricas_filtradas["status"] == "CRÍTICO"])
itens_alerta = len(metricas_filtradas[metricas_filtradas["status"] == "ALERTA"])
estoque_total = metricas_filtradas["estoque_atual"].sum()
media_dias = metricas_filtradas["dias_estoque"].mean()

custos = calcular_relatorio_custos(metricas_filtradas)
custo_total = custos["resumo"]["custo_total_estimado"]

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(criar_kpi_card(
        f"{total_itens:,}",
        "Total de Itens",
        cor="blue"
    ), unsafe_allow_html=True)

with col2:
    st.markdown(criar_kpi_card(
        f"{itens_criticos}",
        "Itens Críticos",
        delta="⚠️ Requer ação" if itens_criticos > 0 else "✓ OK",
        delta_color="negative" if itens_criticos > 0 else "positive",
        cor="red"
    ), unsafe_allow_html=True)

with col3:
    st.markdown(criar_kpi_card(
        f"{itens_alerta}",
        "Em Alerta",
        cor="orange"
    ), unsafe_allow_html=True)

with col4:
    st.markdown(criar_kpi_card(
        f"{estoque_total:,.0f}",
        "Unidades em Estoque",
        cor="green"
    ), unsafe_allow_html=True)

with col5:
    st.markdown(criar_kpi_card(
        f"{media_dias:.1f}",
        "Média Dias Estoque",
        cor="purple"
    ), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================================
# TABS PRINCIPAIS
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Visão Geral",
    "🏪 Por Loja",
    "📦 Reposição",
    "🔮 Previsão",
    "⚡ Otimização"
])


# ============================================================================
# TAB 1 - VISÃO GERAL
# ============================================================================

with tab1:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        # Gráfico de status por loja
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Status de Estoque por Loja</div>', unsafe_allow_html=True)

        status_por_loja = metricas_filtradas.groupby(["loja_id", "status"]).size().unstack(fill_value=0)

        # Reordenar colunas
        ordem_status = ["CRÍTICO", "ALERTA", "ADEQUADO", "EXCESSO"]
        colunas_presentes = [c for c in ordem_status if c in status_por_loja.columns]
        status_por_loja = status_por_loja[colunas_presentes]

        cores_status = {
            "CRÍTICO": "#e53e3e",
            "ALERTA": "#dd6b20",
            "ADEQUADO": "#38a169",
            "EXCESSO": "#3182ce"
        }

        fig_status = go.Figure()
        for status in colunas_presentes:
            fig_status.add_trace(go.Bar(
                name=status,
                x=status_por_loja.index,
                y=status_por_loja[status],
                marker_color=cores_status[status]
            ))

        fig_status.update_layout(
            barmode='stack',
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="",
            yaxis_title="Quantidade de Produtos",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_status, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Gráfico de tendência de vendas
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Tendência de Vendas (Últimos {} dias)</div>'.format(dias_historico), unsafe_allow_html=True)

        vendas_diarias = historico_filtrado.groupby("data")["quantidade_vendida"].sum().reset_index()
        vendas_diarias["media_movel"] = vendas_diarias["quantidade_vendida"].rolling(7, min_periods=1).mean()

        fig_tendencia = go.Figure()

        fig_tendencia.add_trace(go.Scatter(
            x=vendas_diarias["data"],
            y=vendas_diarias["quantidade_vendida"],
            mode='lines',
            name='Vendas Diárias',
            line=dict(color='#90cdf4', width=1),
            fill='tozeroy',
            fillcolor='rgba(144, 205, 244, 0.2)'
        ))

        fig_tendencia.add_trace(go.Scatter(
            x=vendas_diarias["data"],
            y=vendas_diarias["media_movel"],
            mode='lines',
            name='Média Móvel (7 dias)',
            line=dict(color='#3182ce', width=3)
        ))

        fig_tendencia.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="",
            yaxis_title="Unidades Vendidas",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_tendencia, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        # Distribuição de status
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Distribuição de Status</div>', unsafe_allow_html=True)

        status_counts = metricas_filtradas["status"].value_counts()

        fig_pizza = go.Figure(data=[go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            hole=0.6,
            marker_colors=[cores_status.get(s, "#718096") for s in status_counts.index],
            textinfo='percent+label',
            textposition='outside'
        )])

        fig_pizza.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
            annotations=[dict(text=f'{total_itens}', x=0.5, y=0.5, font_size=24, showarrow=False)]
        )

        st.plotly_chart(fig_pizza, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Custos
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Análise de Custos (30 dias)</div>', unsafe_allow_html=True)

        st.metric(
            "Custo de Manutenção",
            formatar_moeda(custos["resumo"]["custo_manutencao_total"])
        )

        st.metric(
            "Risco de Stockout",
            formatar_moeda(custos["resumo"]["risco_stockout_total"])
        )

        st.markdown("---")

        st.metric(
            "**Custo Total Estimado**",
            formatar_moeda(custos["resumo"]["custo_total_estimado"])
        )

        st.markdown('</div>', unsafe_allow_html=True)

        # Alertas
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🚨 Alertas Ativos</div>', unsafe_allow_html=True)

        alertas = gerar_alertas_estoque(metricas_filtradas)

        if len(alertas) > 0:
            for _, alerta in alertas.head(5).iterrows():
                cor_class = "alert-critical" if alerta["status"] == "CRÍTICO" else "alert-warning"
                st.markdown(f"""
                <div class="alert-box {cor_class}">
                    <span>{'🔴' if alerta['status'] == 'CRÍTICO' else '🟠'}</span>
                    <span><strong>{alerta['loja_id']}</strong> - {alerta['produto_nome'][:20]}<br>
                    <small>{alerta['dias_estoque']:.1f} dias de estoque</small></span>
                </div>
                """, unsafe_allow_html=True)

            if len(alertas) > 5:
                st.caption(f"... e mais {len(alertas) - 5} alertas")
        else:
            st.success("✅ Nenhum alerta ativo")

        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# TAB 2 - POR LOJA
# ============================================================================

with tab2:
    col1, col2 = st.columns([1, 2])

    with col1:
        loja_selecionada = st.selectbox(
            "Selecione uma loja",
            options=lojas_selecionadas,
            format_func=lambda x: f"🏪 {x}"
        )

    with col2:
        # Mini KPIs da loja
        metricas_loja = metricas_filtradas[metricas_filtradas["loja_id"] == loja_selecionada]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Produtos", len(metricas_loja))
        c2.metric("Críticos", len(metricas_loja[metricas_loja["status"] == "CRÍTICO"]))
        c3.metric("Estoque Total", f"{metricas_loja['estoque_atual'].sum():,.0f}")
        c4.metric("Dias Médio", f"{metricas_loja['dias_estoque'].mean():.1f}")

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        # Estoque por produto na loja
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">Estoque por Produto - {loja_selecionada}</div>', unsafe_allow_html=True)

        fig_loja = go.Figure()

        fig_loja.add_trace(go.Bar(
            x=metricas_loja["produto_nome"],
            y=metricas_loja["estoque_atual"],
            name="Estoque Atual",
            marker_color=[get_cor_status(s) for s in metricas_loja["status"]]
        ))

        fig_loja.add_trace(go.Scatter(
            x=metricas_loja["produto_nome"],
            y=metricas_loja["ponto_reorder"],
            mode='markers+lines',
            name="Ponto de Reorder",
            line=dict(color='#e53e3e', dash='dash'),
            marker=dict(size=8)
        ))

        fig_loja.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=20, b=80),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis_tickangle=-45,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_loja, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        # Vendas da loja ao longo do tempo
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">Vendas Diárias - {loja_selecionada}</div>', unsafe_allow_html=True)

        vendas_loja = historico_filtrado[historico_filtrado["loja_id"] == loja_selecionada]
        vendas_loja_dia = vendas_loja.groupby("data")["quantidade_vendida"].sum().reset_index()

        fig_vendas_loja = px.area(
            vendas_loja_dia,
            x="data",
            y="quantidade_vendida",
            color_discrete_sequence=["#3182ce"]
        )

        fig_vendas_loja.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="",
            yaxis_title="Unidades",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_vendas_loja, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Tabela detalhada
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">Detalhamento de Produtos - {loja_selecionada}</div>', unsafe_allow_html=True)

    df_display = metricas_loja[[
        "produto_nome", "estoque_atual", "demanda_media", "dias_estoque",
        "estoque_seguranca", "ponto_reorder", "status"
    ]].copy()

    df_display.columns = ["Produto", "Estoque", "Demanda/dia", "Dias Est.", "Est. Seg.", "Ponto Reorder", "Status"]
    df_display = df_display.round(1)

    st.dataframe(
        df_display.style.applymap(
            lambda x: f"background-color: {get_cor_status(x)}20" if x in ["CRÍTICO", "ALERTA", "ADEQUADO", "EXCESSO"] else "",
            subset=["Status"]
        ),
        use_container_width=True,
        hide_index=True
    )

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# TAB 3 - REPOSIÇÃO
# ============================================================================

with tab3:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Recomendações de Reposição Priorizadas</div>', unsafe_allow_html=True)

    recomendacoes = gerar_recomendacoes_reposicao(metricas_filtradas)
    df_recomendacoes = recomendacoes_para_dataframe(recomendacoes)

    if len(df_recomendacoes) > 0:
        # Métricas resumo
        col1, col2, col3 = st.columns(3)

        urgentes = len(df_recomendacoes[df_recomendacoes["prioridade"] == "URGENTE"])
        alta = len(df_recomendacoes[df_recomendacoes["prioridade"] == "ALTA"])
        total_qtd = df_recomendacoes["quantidade_recomendada"].sum()

        col1.metric("🔴 Urgentes", urgentes)
        col2.metric("🟠 Alta Prioridade", alta)
        col3.metric("📦 Total a Repor", f"{total_qtd:,.0f} un")

        st.markdown("---")

        # Filtro de prioridade
        prioridades = st.multiselect(
            "Filtrar por prioridade",
            options=["URGENTE", "ALTA", "MEDIA", "BAIXA"],
            default=["URGENTE", "ALTA"]
        )

        df_filtrado = df_recomendacoes[df_recomendacoes["prioridade"].isin(prioridades)]

        # Tabela de recomendações
        df_display = df_filtrado[[
            "prioridade", "loja_id", "produto_nome", "quantidade_recomendada",
            "dias_ate_stockout", "custo_estimado_stockout"
        ]].copy()

        df_display.columns = ["Prioridade", "Loja", "Produto", "Qtd. Recomendada", "Dias até Stockout", "Custo Stockout Est."]

        # Cores por prioridade
        cores_prioridade = {
            "URGENTE": "#fed7d7",
            "ALTA": "#feebc8",
            "MEDIA": "#fefcbf",
            "BAIXA": "#c6f6d5"
        }

        def highlight_prioridade(row):
            cor = cores_prioridade.get(row["Prioridade"], "")
            return [f"background-color: {cor}"] * len(row)

        st.dataframe(
            df_display.style.apply(highlight_prioridade, axis=1),
            use_container_width=True,
            hide_index=True,
            height=400
        )

        # Download
        csv = df_filtrado.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Exportar Recomendações (CSV)",
            csv,
            "recomendacoes_reposicao.csv",
            "text/csv"
        )
    else:
        st.success("✅ Nenhuma reposição necessária no momento!")

    st.markdown('</div>', unsafe_allow_html=True)

    # Simulação What-If
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔄 Simulação What-If</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        fator_demanda = st.slider(
            "Variação na demanda (%)",
            min_value=-50,
            max_value=100,
            value=0,
            step=10,
            format="%d%%"
        )

        fator = 1 + (fator_demanda / 100)

        st.info(f"Simulando demanda {'aumentada' if fator > 1 else 'reduzida'} em {abs(fator_demanda)}%")

    with col2:
        if fator != 1:
            simulacao = simular_cenario_what_if(metricas_filtradas, fator_demanda=fator)

            # Comparar status
            status_original = metricas_filtradas.groupby("status").size()
            status_simulado = simulacao.groupby("status_simulado").size()

            comparacao = pd.DataFrame({
                "Atual": status_original,
                "Simulado": status_simulado
            }).fillna(0).astype(int)

            comparacao["Diferença"] = comparacao["Simulado"] - comparacao["Atual"]

            fig_comparacao = go.Figure()

            fig_comparacao.add_trace(go.Bar(
                name='Atual',
                x=comparacao.index,
                y=comparacao['Atual'],
                marker_color='#90cdf4'
            ))

            fig_comparacao.add_trace(go.Bar(
                name='Simulado',
                x=comparacao.index,
                y=comparacao['Simulado'],
                marker_color='#3182ce'
            ))

            fig_comparacao.update_layout(
                barmode='group',
                height=250,
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig_comparacao, use_container_width=True)
        else:
            st.info("Ajuste o slider para simular cenários de demanda")

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# TAB 4 - PREVISÃO
# ============================================================================

with tab4:
    if not FORECASTING_AVAILABLE:
        st.warning("⚠️ Módulo de previsão não disponível. Instale: `pip install statsmodels`")
    else:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔮 Previsão de Demanda (Exponential Smoothing)</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            horizonte = st.slider("Horizonte de previsão (dias)", 7, 30, 14)

        with col2:
            metodo = st.selectbox(
                "Método",
                options=["auto", "ses", "holt", "holt_winters"],
                format_func=lambda x: {
                    "auto": "🎯 Automático (melhor MAPE)",
                    "ses": "Simple Exponential Smoothing",
                    "holt": "Holt Linear Trend",
                    "holt_winters": "Holt-Winters (Sazonal)"
                }[x]
            )

        with col3:
            produto_prev = st.selectbox(
                "Produto",
                options=["TODOS"] + produtos_selecionados,
                format_func=lambda x: "📦 Todos os produtos" if x == "TODOS" else PRODUTOS[x]["nome"]
            )

        if st.button("🚀 Gerar Previsão", type="primary"):
            with st.spinner("Calculando previsões..."):
                previsor = PrevisaoDemanda(historico_filtrado)

                if produto_prev == "TODOS":
                    resultados = previsor.prever_todos_produtos(horizonte, metodo)
                else:
                    if metodo == "auto":
                        resultado = previsor.auto_selecionar_metodo(produto_prev, None, horizonte)
                    elif metodo == "holt_winters":
                        resultado = previsor.holt_winters(produto_prev, None, horizonte)
                    elif metodo == "holt":
                        resultado = previsor.holt_linear(produto_prev, None, horizonte)
                    else:
                        resultado = previsor.simple_exponential_smoothing(produto_prev, None, horizonte)
                    resultados = {produto_prev: resultado}

                # Exibir resultados
                for prod_id, res in resultados.items():
                    nome = PRODUTOS[prod_id]["nome"]

                    col_info, col_chart = st.columns([1, 2])

                    with col_info:
                        st.markdown(f"### {nome}")
                        st.caption(f"Método: {res.metodo}")

                        m1, m2, m3 = st.columns(3)
                        m1.metric("MAPE", f"{res.metricas['mape']:.1f}%")
                        m2.metric("MAE", f"{res.metricas['mae']:.0f}")
                        m3.metric("RMSE", f"{res.metricas['rmse']:.0f}")

                        total = res.previsoes["previsao"].sum()
                        st.metric("Total Previsto", f"{total:,.0f} un")

                    with col_chart:
                        # Série histórica + previsão
                        serie = previsor.preparar_serie_temporal(prod_id)

                        fig_prev = go.Figure()

                        # Histórico
                        fig_prev.add_trace(go.Scatter(
                            x=serie.index,
                            y=serie.values,
                            mode='lines',
                            name='Histórico',
                            line=dict(color='#3182ce')
                        ))

                        # Previsão
                        fig_prev.add_trace(go.Scatter(
                            x=res.previsoes["data"],
                            y=res.previsoes["previsao"],
                            mode='lines+markers',
                            name='Previsão',
                            line=dict(color='#38a169', dash='dash')
                        ))

                        # Intervalo de confiança
                        fig_prev.add_trace(go.Scatter(
                            x=pd.concat([res.previsoes["data"], res.previsoes["data"][::-1]]),
                            y=pd.concat([res.previsoes["limite_superior"], res.previsoes["limite_inferior"][::-1]]),
                            fill='toself',
                            fillcolor='rgba(56, 161, 105, 0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='IC 95%'
                        ))

                        fig_prev.update_layout(
                            height=300,
                            margin=dict(l=20, r=20, t=20, b=20),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )

                        st.plotly_chart(fig_prev, use_container_width=True)

                    st.markdown("---")

                # Export
                df_previsoes = previsoes_para_dataframe(resultados)
                csv = df_previsoes.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Exportar Previsões (CSV)",
                    csv,
                    "previsao_demanda.csv",
                    "text/csv"
                )

        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# TAB 5 - OTIMIZAÇÃO
# ============================================================================

with tab5:
    if not LINEAR_OPT_AVAILABLE:
        st.warning("⚠️ Módulo de otimização não disponível. Instale: `pip install pulp`")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📊 Otimização de Alocação</div>', unsafe_allow_html=True)

            nivel_servico = st.slider(
                "Nível de Serviço Desejado",
                min_value=0.90,
                max_value=0.99,
                value=0.95,
                step=0.01,
                format="%.0f%%"
            )

            if st.button("🎯 Otimizar Alocação", type="primary", key="btn_alocacao"):
                with st.spinner("Executando otimização..."):
                    otimizador = OtimizadorLinear(metricas_filtradas)

                    # Estoque disponível no CD
                    estoque_cd = {
                        prod: int(metricas_filtradas[
                            metricas_filtradas["produto_id"] == prod
                        ]["demanda_media"].sum() * 30)
                        for prod in PRODUTOS.keys()
                    }

                    resultado = otimizador.otimizar_alocacao_estoque(
                        estoque_cd,
                        nivel_servico_minimo=nivel_servico
                    )

                    st.success(f"✅ {resultado.status} - {resultado.mensagem}")

                    m1, m2 = st.columns(2)
                    m1.metric("Custo Total Otimizado", formatar_moeda(resultado.custo_total))
                    m2.metric("Tempo de Execução", f"{resultado.tempo_execucao}s")

                    if len(resultado.alocacoes) > 0:
                        st.dataframe(
                            resultado.alocacoes.head(10),
                            use_container_width=True,
                            hide_index=True
                        )

            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">💰 Otimização com Orçamento</div>', unsafe_allow_html=True)

            orcamento = st.number_input(
                "Orçamento Disponível (R$)",
                min_value=10000,
                max_value=500000,
                value=50000,
                step=5000
            )

            if st.button("💵 Otimizar Compras", type="primary", key="btn_orcamento"):
                with st.spinner("Executando otimização..."):
                    otimizador = OtimizadorLinear(metricas_filtradas)

                    precos = {
                        prod: config["preco_unitario"] * 0.6
                        for prod, config in PRODUTOS.items()
                    }

                    resultado = otimizador.otimizar_reposicao_orcamento(
                        orcamento=orcamento,
                        precos_compra=precos
                    )

                    st.success(f"✅ {resultado.status}")
                    st.info(resultado.mensagem)

                    if len(resultado.alocacoes) > 0:
                        # Gráfico de compras
                        fig_compras = px.bar(
                            resultado.alocacoes,
                            x="produto_nome",
                            y="quantidade_compra",
                            color="custo_total",
                            color_continuous_scale="Blues"
                        )

                        fig_compras.update_layout(
                            height=300,
                            margin=dict(l=20, r=20, t=20, b=80),
                            xaxis_tickangle=-45
                        )

                        st.plotly_chart(fig_compras, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #718096; font-size: 0.875rem; padding: 1rem;">
    <strong>Inventory Optimizer</strong> • Sistema de Otimização de Estoque<br>
    Desenvolvido com Streamlit + Plotly • {} itens monitorados
</div>
""".format(len(metricas)), unsafe_allow_html=True)
