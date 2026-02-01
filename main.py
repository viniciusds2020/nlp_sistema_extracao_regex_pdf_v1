#!/usr/bin/env python3
"""
Sistema de Otimização de Estoque
================================
Sistema completo para otimização de estoque de um centro de distribuição
que abastece 10 lojas com diferentes padrões de vendas.

Funcionalidades:
1. Geração de dados sintéticos realistas
2. Análise estatística de demanda e sazonalidade
3. Cálculo de estoque de segurança e ponto de reorder
4. Recomendações automáticas de reposição
5. Alertas de estoque crítico
6. Simulação de cenários what-if
7. Dashboard visual e relatórios
8. Otimização linear para alocação de estoque e orçamento
9. Previsão de demanda com Exponential Smoothing (Holt-Winters)

Uso:
    python main.py [--dias DIAS] [--cenario CENARIO] [--output PASTA] [--otimizar]

Argumentos:
    --dias: Número de dias de histórico a analisar (padrão: 30)
    --cenario: Fator de demanda para simulação what-if (ex: 1.3 para +30%)
    --output: Pasta para salvar relatórios (padrão: reports)
    --otimizar: Executar otimização linear de alocação
    --orcamento: Orçamento para otimização de compras (ex: 50000)
    --prever: Gerar previsões de demanda para N dias (ex: 7)

Autor: Sistema de Otimização de Estoque
Data: 2024
"""

from typing import TYPE_CHECKING
import argparse
import os
import sys
from datetime import datetime

import pandas as pd
import numpy as np

# Adiciona o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_generator import (
    gerar_historico_vendas,
    gerar_estoque_atual,
    get_produtos_info,
    get_lojas_info,
    LOJAS,
    PRODUTOS
)

from statistical_analysis import (
    calcular_estatisticas_demanda,
    analisar_sazonalidade_semanal,
    calcular_metricas_estoque,
    identificar_tendencia,
    calcular_variabilidade_demanda
)

from inventory_optimizer import (
    gerar_recomendacoes_reposicao,
    gerar_alertas_estoque,
    simular_cenario_what_if,
    calcular_relatorio_custos,
    recomendacoes_para_dataframe
)

from visualization import (
    criar_dashboard_estoque,
    plotar_tendencia_demanda,
    plotar_comparativo_lojas,
    gerar_relatorio_texto,
    plotar_todos_graficos
)

# Importação condicional do módulo de otimização linear
try:
    from linear_optimization import (
        OtimizadorLinear,
        otimizar_mix_produtos,
        gerar_relatorio_otimizacao,
        ResultadoOtimizacao,
        PULP_AVAILABLE
    )
    LINEAR_OPT_AVAILABLE = PULP_AVAILABLE
except ImportError:
    LINEAR_OPT_AVAILABLE = False

# Importação condicional do módulo de previsão
try:
    from forecasting import (
        PrevisaoDemanda,
        gerar_relatorio_previsao,
        previsoes_para_dataframe,
        ResultadoPrevisao,
        STATSMODELS_AVAILABLE
    )
    FORECASTING_AVAILABLE = STATSMODELS_AVAILABLE
except ImportError:
    FORECASTING_AVAILABLE = False


class SistemaOtimizacaoEstoque:
    """
    Classe principal que orquestra todas as funcionalidades do sistema.
    """

    def __init__(self, dias_historico: int = 30, seed: int = 42):
        """
        Inicializa o sistema com dados sintéticos.

        Args:
            dias_historico: Número de dias de histórico a gerar
            seed: Seed para reprodutibilidade
        """
        self.dias_historico = dias_historico
        self.seed = seed

        # DataFrames principais
        self.historico_vendas = None
        self.estoque_atual = None
        self.stats_demanda = None
        self.metricas_estoque = None

        # Informações de referência
        self.produtos_info = get_produtos_info()
        self.lojas_info = get_lojas_info()

        # Flag de inicialização
        self._inicializado = False

    def inicializar(self) -> None:
        """
        Gera dados e calcula todas as métricas necessárias.
        """
        print("=" * 60)
        print("   SISTEMA DE OTIMIZAÇÃO DE ESTOQUE")
        print("=" * 60)
        print(f"\nInicializando sistema...")
        print(f"- Período de análise: {self.dias_historico} dias")
        print(f"- Lojas monitoradas: {len(LOJAS)}")
        print(f"- Produtos cadastrados: {len(PRODUTOS)}")

        # Gerar dados
        print("\n[1/4] Gerando histórico de vendas...")
        self.historico_vendas = gerar_historico_vendas(
            dias=self.dias_historico,
            seed=self.seed
        )
        print(f"      {len(self.historico_vendas):,} registros gerados")

        # Gerar estoque atual
        print("[2/4] Gerando níveis de estoque atual...")
        self.estoque_atual = gerar_estoque_atual(
            self.historico_vendas,
            seed=self.seed
        )

        # Calcular estatísticas
        print("[3/4] Calculando estatísticas de demanda...")
        self.stats_demanda = calcular_estatisticas_demanda(self.historico_vendas)

        # Calcular métricas de estoque
        print("[4/4] Calculando métricas de estoque...")
        self.metricas_estoque = calcular_metricas_estoque(
            self.estoque_atual,
            self.stats_demanda
        )

        self._inicializado = True
        print("\n✓ Sistema inicializado com sucesso!")

    def obter_dashboard_resumo(self) -> dict:
        """
        Retorna resumo do status atual do estoque.

        Returns:
            Dict com métricas resumidas
        """
        self._verificar_inicializacao()

        df = self.metricas_estoque

        return {
            "total_itens": len(df),
            "criticos": len(df[df["status"] == "CRÍTICO"]),
            "alerta": len(df[df["status"] == "ALERTA"]),
            "adequado": len(df[df["status"] == "ADEQUADO"]),
            "excesso": len(df[df["status"] == "EXCESSO"]),
            "estoque_total": int(df["estoque_atual"].sum()),
            "media_dias_estoque": round(df["dias_estoque"].mean(), 1),
            "data_analise": datetime.now().strftime("%d/%m/%Y %H:%M")
        }

    def obter_recomendacoes(self, top_n: int = 20) -> pd.DataFrame:
        """
        Obtém recomendações de reposição priorizadas.

        Args:
            top_n: Número máximo de recomendações a retornar

        Returns:
            DataFrame com recomendações
        """
        self._verificar_inicializacao()

        recomendacoes = gerar_recomendacoes_reposicao(self.metricas_estoque)
        df = recomendacoes_para_dataframe(recomendacoes)

        return df.head(top_n)

    def obter_alertas(self) -> pd.DataFrame:
        """
        Obtém alertas de estoque crítico.

        Returns:
            DataFrame com alertas
        """
        self._verificar_inicializacao()
        return gerar_alertas_estoque(self.metricas_estoque)

    def simular_cenario(self, fator_demanda: float) -> pd.DataFrame:
        """
        Simula cenário what-if com alteração na demanda.

        Args:
            fator_demanda: Multiplicador de demanda (1.2 = +20%)

        Returns:
            DataFrame com resultados da simulação
        """
        self._verificar_inicializacao()

        return simular_cenario_what_if(
            self.metricas_estoque,
            fator_demanda=fator_demanda
        )

    def obter_relatorio_custos(self, horizonte_dias: int = 30) -> dict:
        """
        Obtém relatório detalhado de custos.

        Args:
            horizonte_dias: Horizonte de análise em dias

        Returns:
            Dict com relatório de custos
        """
        self._verificar_inicializacao()

        return calcular_relatorio_custos(
            self.metricas_estoque,
            horizonte_dias=horizonte_dias
        )

    def obter_analise_produto(self, produto_id: str) -> dict:
        """
        Obtém análise detalhada de um produto.

        Args:
            produto_id: ID do produto (ex: "PROD_001")

        Returns:
            Dict com análise do produto
        """
        self._verificar_inicializacao()

        # Tendência
        tendencia = identificar_tendencia(self.historico_vendas, produto_id)

        # Sazonalidade
        sazonalidade = analisar_sazonalidade_semanal(self.historico_vendas)
        sazonalidade_produto = sazonalidade[
            sazonalidade["produto_id"] == produto_id
        ].to_dict(orient="records")

        # Métricas por loja
        metricas_produto = self.metricas_estoque[
            self.metricas_estoque["produto_id"] == produto_id
        ].to_dict(orient="records")

        return {
            "produto_id": produto_id,
            "tendencia": tendencia,
            "sazonalidade_semanal": sazonalidade_produto,
            "metricas_por_loja": metricas_produto
        }

    def gerar_relatorios(self, pasta_saida: str = "reports") -> list:
        """
        Gera todos os relatórios e gráficos.

        Args:
            pasta_saida: Pasta para salvar arquivos

        Returns:
            Lista de arquivos gerados
        """
        self._verificar_inicializacao()

        os.makedirs(pasta_saida, exist_ok=True)
        arquivos = []

        print(f"\nGerando relatórios em '{pasta_saida}/'...")

        # 1. Gráficos
        print("- Gerando gráficos...")
        graficos = plotar_todos_graficos(
            self.historico_vendas,
            self.metricas_estoque,
            pasta_saida
        )
        arquivos.extend(graficos)

        # 2. Relatório de texto
        print("- Gerando relatório de texto...")
        recomendacoes = self.obter_recomendacoes(top_n=50)
        custos = self.obter_relatorio_custos()

        path_relatorio = os.path.join(pasta_saida, "relatorio_estoque.txt")
        gerar_relatorio_texto(
            self.metricas_estoque,
            recomendacoes,
            custos,
            salvar_path=path_relatorio
        )
        arquivos.append(path_relatorio)

        # 3. Exportar dados para CSV
        print("- Exportando dados para CSV...")

        # Métricas de estoque
        path_metricas = os.path.join(pasta_saida, "metricas_estoque.csv")
        self.metricas_estoque.to_csv(path_metricas, index=False)
        arquivos.append(path_metricas)

        # Recomendações
        path_recomendacoes = os.path.join(pasta_saida, "recomendacoes.csv")
        recomendacoes.to_csv(path_recomendacoes, index=False)
        arquivos.append(path_recomendacoes)

        # Alertas
        alertas = self.obter_alertas()
        if len(alertas) > 0:
            path_alertas = os.path.join(pasta_saida, "alertas.csv")
            alertas.to_csv(path_alertas, index=False)
            arquivos.append(path_alertas)

        print(f"\n✓ {len(arquivos)} arquivos gerados!")
        return arquivos

    def exibir_resumo_console(self) -> None:
        """
        Exibe resumo completo no console.
        """
        self._verificar_inicializacao()

        print("\n" + "=" * 60)
        print("   RESUMO DO SISTEMA DE ESTOQUE")
        print("=" * 60)

        # Dashboard resumo
        resumo = self.obter_dashboard_resumo()
        print(f"\n📊 STATUS GERAL ({resumo['data_analise']})")
        print("-" * 40)
        print(f"   Total de itens: {resumo['total_itens']}")
        print(f"   🔴 Críticos:     {resumo['criticos']}")
        print(f"   🟡 Em Alerta:    {resumo['alerta']}")
        print(f"   🟢 Adequados:    {resumo['adequado']}")
        print(f"   🔵 Em Excesso:   {resumo['excesso']}")
        print(f"\n   Estoque Total: {resumo['estoque_total']:,} unidades")
        print(f"   Média de Cobertura: {resumo['media_dias_estoque']} dias")

        # Custos
        custos = self.obter_relatorio_custos()
        print(f"\n💰 ANÁLISE DE CUSTOS (30 dias)")
        print("-" * 40)
        print(f"   Custo Manutenção:  R$ {custos['resumo']['custo_manutencao_total']:>12,.2f}")
        print(f"   Risco Stockout:    R$ {custos['resumo']['risco_stockout_total']:>12,.2f}")
        print(f"   {'─' * 36}")
        print(f"   TOTAL ESTIMADO:    R$ {custos['resumo']['custo_total_estimado']:>12,.2f}")

        # Top 5 recomendações
        recomendacoes = self.obter_recomendacoes(top_n=5)
        if len(recomendacoes) > 0:
            print(f"\n📋 TOP 5 REPOSIÇÕES URGENTES")
            print("-" * 40)
            for _, row in recomendacoes.iterrows():
                print(f"   [{row['prioridade'][:3]}] {row['loja_id']} - {row['produto_nome'][:20]}")
                print(f"         Qtd: {row['quantidade_recomendada']} | {row['dias_ate_stockout']:.1f} dias até stockout")

        # Alertas
        alertas = self.obter_alertas()
        n_alertas = len(alertas)
        if n_alertas > 0:
            print(f"\n⚠️  {n_alertas} ALERTAS ATIVOS")
            print("-" * 40)
            for _, row in alertas.head(5).iterrows():
                print(f"   {row['mensagem'][:55]}")

        print("\n" + "=" * 60)

    def otimizar_alocacao(
        self,
        estoque_cd: dict = None,
        nivel_servico: float = 0.95
    ) -> 'ResultadoOtimizacao':
        """
        Executa otimização linear para alocação de estoque.

        Distribui o estoque do Centro de Distribuição para as lojas
        minimizando custo total (manutenção + risco de stockout).

        Args:
            estoque_cd: Dict com estoque disponível por produto no CD
            nivel_servico: Nível de serviço mínimo (0-1)

        Returns:
            ResultadoOtimizacao com alocações ótimas
        """
        self._verificar_inicializacao()

        if not LINEAR_OPT_AVAILABLE:
            raise ImportError(
                "PuLP não está instalado. Instale com: pip install pulp"
            )

        # Estoque padrão se não fornecido
        if estoque_cd is None:
            estoque_cd = {
                prod: int(self.metricas_estoque[
                    self.metricas_estoque["produto_id"] == prod
                ]["demanda_media"].sum() * 30)
                for prod in PRODUTOS.keys()
            }

        otimizador = OtimizadorLinear(self.metricas_estoque)
        return otimizador.otimizar_alocacao_estoque(
            estoque_disponivel=estoque_cd,
            nivel_servico_minimo=nivel_servico
        )

    def otimizar_compras_orcamento(
        self,
        orcamento: float,
        horizonte_dias: int = 30
    ) -> 'ResultadoOtimizacao':
        """
        Otimiza compras com orçamento limitado.

        Decide quanto comprar de cada produto respeitando o orçamento,
        minimizando o risco de stockout.

        Args:
            orcamento: Orçamento total disponível
            horizonte_dias: Horizonte de planejamento em dias

        Returns:
            ResultadoOtimizacao com quantidades ótimas de compra
        """
        self._verificar_inicializacao()

        if not LINEAR_OPT_AVAILABLE:
            raise ImportError(
                "PuLP não está instalado. Instale com: pip install pulp"
            )

        # Preços de compra (60% do preço de venda)
        precos = {
            prod: config["preco_unitario"] * 0.6
            for prod, config in PRODUTOS.items()
        }

        otimizador = OtimizadorLinear(self.metricas_estoque)
        return otimizador.otimizar_reposicao_orcamento(
            orcamento=orcamento,
            precos_compra=precos,
            horizonte_dias=horizonte_dias
        )

    def otimizar_mix_produtos(
        self,
        espaco_total: float = 5000
    ) -> 'ResultadoOtimizacao':
        """
        Otimiza o mix de produtos considerando espaço e margem.

        Args:
            espaco_total: Espaço total disponível em m³

        Returns:
            ResultadoOtimizacao com quantidades ótimas
        """
        self._verificar_inicializacao()

        if not LINEAR_OPT_AVAILABLE:
            raise ImportError(
                "PuLP não está instalado. Instale com: pip install pulp"
            )

        # Espaço por unidade (estimado)
        espaco = {prod: 0.05 for prod in PRODUTOS.keys()}

        # Margem por produto (30% do preço)
        margem = {
            prod: config["preco_unitario"] * 0.3
            for prod, config in PRODUTOS.items()
        }

        return otimizar_mix_produtos(
            self.metricas_estoque,
            espaco_total=espaco_total,
            espaco_por_unidade=espaco,
            margem_por_produto=margem
        )

    def exibir_resultado_otimizacao(self, resultado: 'ResultadoOtimizacao') -> None:
        """
        Exibe os resultados da otimização no console.

        Args:
            resultado: Objeto ResultadoOtimizacao
        """
        print("\n" + "=" * 60)
        print("   RESULTADO DA OTIMIZAÇÃO LINEAR")
        print("=" * 60)
        print(f"\n   Status: {resultado.status}")
        print(f"   Tempo de execução: {resultado.tempo_execucao}s")
        print(f"   {resultado.mensagem}")

        print(f"\n   💰 CUSTOS OTIMIZADOS")
        print("   " + "-" * 40)
        print(f"   Custo Total:      R$ {resultado.custo_total:>12,.2f}")
        if resultado.custo_manutencao > 0:
            print(f"   Custo Manutenção: R$ {resultado.custo_manutencao:>12,.2f}")
        if resultado.custo_stockout > 0:
            print(f"   Custo Stockout:   R$ {resultado.custo_stockout:>12,.2f}")

        if len(resultado.alocacoes) > 0:
            print(f"\n   📦 TOP 10 ALOCAÇÕES/RECOMENDAÇÕES")
            print("   " + "-" * 40)

            # Mostrar primeiras 10 linhas
            for i, (_, row) in enumerate(resultado.alocacoes.head(10).iterrows()):
                if "quantidade_alocada" in row:
                    print(f"   {row.get('loja_id', '')} - {row.get('produto_nome', row.get('produto_id', ''))[:20]}")
                    print(f"      Alocar: {int(row['quantidade_alocada'])} unidades")
                elif "quantidade_compra" in row:
                    print(f"   {row.get('produto_nome', row.get('produto_id', ''))[:25]}")
                    print(f"      Comprar: {int(row['quantidade_compra'])} un | R$ {row.get('custo_total', 0):,.2f}")
                elif "quantidade_otima" in row:
                    print(f"   {row.get('produto_nome', row.get('produto_id', ''))[:25]}")
                    print(f"      Quantidade: {int(row['quantidade_otima'])} | Margem: R$ {row.get('margem_total', 0):,.2f}")

            if len(resultado.alocacoes) > 10:
                print(f"\n   ... e mais {len(resultado.alocacoes) - 10} itens")

        print("\n" + "=" * 60)

    def prever_demanda(
        self,
        horizonte: int = 7,
        metodo: str = "auto",
        produto_id: str = None
    ) -> dict:
        """
        Gera previsões de demanda usando Exponential Smoothing.

        Args:
            horizonte: Número de dias a prever
            metodo: 'auto', 'ses', 'holt', 'holt_winters'
            produto_id: ID do produto (None para todos)

        Returns:
            Dict com resultados de previsão
        """
        self._verificar_inicializacao()

        if not FORECASTING_AVAILABLE:
            raise ImportError(
                "statsmodels não está instalado. Instale com: pip install statsmodels"
            )

        previsor = PrevisaoDemanda(self.historico_vendas)

        if produto_id:
            # Previsão para produto específico
            if metodo == "auto":
                resultado = previsor.auto_selecionar_metodo(produto_id, None, horizonte)
            elif metodo == "ses":
                resultado = previsor.simple_exponential_smoothing(produto_id, None, horizonte)
            elif metodo == "holt":
                resultado = previsor.holt_linear(produto_id, None, horizonte)
            elif metodo == "holt_winters":
                resultado = previsor.holt_winters(produto_id, None, horizonte)
            else:
                resultado = previsor.auto_selecionar_metodo(produto_id, None, horizonte)

            return {produto_id: resultado}
        else:
            # Previsão para todos os produtos
            return previsor.prever_todos_produtos(horizonte, metodo)

    def exibir_previsoes(self, resultados: dict) -> None:
        """
        Exibe as previsões no console.

        Args:
            resultados: Dict com ResultadoPrevisao por produto
        """
        print("\n" + "=" * 60)
        print("   PREVISÃO DE DEMANDA (Exponential Smoothing)")
        print("=" * 60)

        for produto_id, resultado in resultados.items():
            nome = PRODUTOS.get(produto_id, {}).get("nome", produto_id)

            print(f"\n   📈 {nome}")
            print("   " + "-" * 40)
            print(f"   Método: {resultado.metodo}")
            print(f"   MAPE: {resultado.metricas['mape']:.1f}%")
            print(f"   MAE: {resultado.metricas['mae']:.1f} unidades")

            print(f"\n   Previsões:")
            total_previsto = 0
            for _, row in resultado.previsoes.iterrows():
                data_str = row["data"].strftime("%d/%m")
                prev = row["previsao"]
                total_previsto += prev
                print(f"      {data_str}: {prev:>6.0f} un [{row['limite_inferior']:.0f}-{row['limite_superior']:.0f}]")

            print(f"\n   Total previsto: {total_previsto:.0f} unidades")

        print("\n" + "=" * 60)

    def _verificar_inicializacao(self) -> None:
        """Verifica se o sistema foi inicializado."""
        if not self._inicializado:
            raise RuntimeError(
                "Sistema não inicializado. Execute inicializar() primeiro."
            )


def main():
    """
    Função principal para execução via linha de comando.
    """
    parser = argparse.ArgumentParser(
        description="Sistema de Otimização de Estoque",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python main.py                          # Execução padrão
  python main.py --dias 60                # Análise de 60 dias
  python main.py --cenario 1.3            # Simular +30% de demanda
  python main.py --output meus_relatorios # Salvar em pasta específica
        """
    )

    parser.add_argument(
        "--dias",
        type=int,
        default=30,
        help="Número de dias de histórico (padrão: 30)"
    )

    parser.add_argument(
        "--cenario",
        type=float,
        default=None,
        help="Fator de demanda para simulação (ex: 1.3 para +30%%)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="reports",
        help="Pasta para salvar relatórios (padrão: reports)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para reprodutibilidade (padrão: 42)"
    )

    parser.add_argument(
        "--no-graficos",
        action="store_true",
        help="Não gerar gráficos (apenas relatórios CSV/TXT)"
    )

    parser.add_argument(
        "--otimizar",
        action="store_true",
        help="Executar otimização linear de alocação de estoque"
    )

    parser.add_argument(
        "--orcamento",
        type=float,
        default=None,
        help="Orçamento para otimização de compras (ex: 50000)"
    )

    parser.add_argument(
        "--otimizar-mix",
        action="store_true",
        help="Otimizar mix de produtos (maximizar margem)"
    )

    parser.add_argument(
        "--prever",
        type=int,
        default=None,
        metavar="DIAS",
        help="Gerar previsão de demanda para N dias (ex: 7)"
    )

    parser.add_argument(
        "--metodo-previsao",
        type=str,
        default="auto",
        choices=["auto", "ses", "holt", "holt_winters"],
        help="Método de previsão: auto, ses, holt, holt_winters (padrão: auto)"
    )

    args = parser.parse_args()

    # Inicializar sistema
    sistema = SistemaOtimizacaoEstoque(
        dias_historico=args.dias,
        seed=args.seed
    )
    sistema.inicializar()

    # Exibir resumo
    sistema.exibir_resumo_console()

    # Simulação what-if se solicitada
    if args.cenario:
        print(f"\n" + "=" * 60)
        print(f"   SIMULAÇÃO WHAT-IF: Demanda x{args.cenario}")
        print("=" * 60)

        simulacao = sistema.simular_cenario(args.cenario)

        # Comparar status
        original = sistema.metricas_estoque.groupby("status").size()
        simulado = simulacao.groupby("status_simulado").size()

        print("\n   Status Atual vs Simulado:")
        print("   " + "-" * 40)
        for status in ["CRÍTICO", "ALERTA", "ADEQUADO", "EXCESSO"]:
            orig = original.get(status, 0)
            sim = simulado.get(status, 0)
            diff = sim - orig
            sinal = "+" if diff > 0 else ""
            print(f"   {status:>10}: {orig:>4} → {sim:>4} ({sinal}{diff})")

        # Impacto nos custos
        custo_original = sistema.metricas_estoque["estoque_atual"].sum() * \
                        sistema.metricas_estoque["custo_manutencao"].mean() * 30
        custo_simulado = simulacao["custo_total_periodo"].sum()

        print(f"\n   Impacto nos Custos:")
        print(f"   Custo Original: R$ {custo_original:,.2f}")
        print(f"   Custo Simulado: R$ {custo_simulado:,.2f}")

    # Otimização linear
    if args.otimizar or args.orcamento or getattr(args, 'otimizar_mix', False):
        if not LINEAR_OPT_AVAILABLE:
            print("\n⚠️  PuLP não está instalado. Instale com: pip install pulp")
        else:
            # Otimização de alocação
            if args.otimizar:
                print(f"\n" + "=" * 60)
                print("   OTIMIZAÇÃO LINEAR: Alocação de Estoque")
                print("=" * 60)

                resultado = sistema.otimizar_alocacao(nivel_servico=0.95)
                sistema.exibir_resultado_otimizacao(resultado)

                # Salvar resultado
                if len(resultado.alocacoes) > 0:
                    path_otim = os.path.join(args.output, "otimizacao_alocacao.csv")
                    resultado.alocacoes.to_csv(path_otim, index=False)
                    print(f"\n   Resultado salvo em: {path_otim}")

            # Otimização com orçamento
            if args.orcamento:
                print(f"\n" + "=" * 60)
                print(f"   OTIMIZAÇÃO LINEAR: Compras com Orçamento R$ {args.orcamento:,.2f}")
                print("=" * 60)

                resultado = sistema.otimizar_compras_orcamento(
                    orcamento=args.orcamento
                )
                sistema.exibir_resultado_otimizacao(resultado)

                # Salvar resultado
                if len(resultado.alocacoes) > 0:
                    path_otim = os.path.join(args.output, "otimizacao_compras.csv")
                    resultado.alocacoes.to_csv(path_otim, index=False)
                    print(f"\n   Resultado salvo em: {path_otim}")

            # Otimização de mix
            if getattr(args, 'otimizar_mix', False):
                print(f"\n" + "=" * 60)
                print("   OTIMIZAÇÃO LINEAR: Mix de Produtos")
                print("=" * 60)

                resultado = sistema.otimizar_mix_produtos(espaco_total=5000)
                sistema.exibir_resultado_otimizacao(resultado)

                # Salvar resultado
                if len(resultado.alocacoes) > 0:
                    path_otim = os.path.join(args.output, "otimizacao_mix.csv")
                    resultado.alocacoes.to_csv(path_otim, index=False)
                    print(f"\n   Resultado salvo em: {path_otim}")

    # Previsão de demanda
    if args.prever:
        if not FORECASTING_AVAILABLE:
            print("\n⚠️  statsmodels não está instalado. Instale com: pip install statsmodels")
        else:
            print(f"\n" + "=" * 60)
            print(f"   PREVISÃO DE DEMANDA: {args.prever} dias")
            print("=" * 60)

            resultados_previsao = sistema.prever_demanda(
                horizonte=args.prever,
                metodo=args.metodo_previsao
            )

            sistema.exibir_previsoes(resultados_previsao)

            # Salvar previsões em CSV
            df_previsoes = previsoes_para_dataframe(resultados_previsao)
            path_previsao = os.path.join(args.output, "previsao_demanda.csv")
            df_previsoes.to_csv(path_previsao, index=False)
            print(f"\n   Previsões salvas em: {path_previsao}")

            # Resumo total
            total_previsto = df_previsoes["previsao"].sum()
            print(f"\n   📊 Demanda total prevista ({args.prever} dias): {total_previsto:,.0f} unidades")

    # Gerar relatórios
    arquivos = sistema.gerar_relatorios(args.output)

    print(f"\n📁 Arquivos gerados:")
    for arq in arquivos:
        print(f"   - {arq}")

    print(f"\n✅ Processo concluído com sucesso!")

    return sistema


if __name__ == "__main__":
    sistema = main()
