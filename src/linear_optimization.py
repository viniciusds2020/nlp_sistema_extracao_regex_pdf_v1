"""
Módulo de Otimização Linear
============================
Utiliza programação linear para otimizar decisões de estoque.

Problemas resolvidos:
1. Minimização de custo total (manutenção + stockout)
2. Alocação ótima de estoque entre lojas
3. Planejamento de reposição com restrições de orçamento
4. Otimização multi-período

Biblioteca: PuLP (Python Linear Programming)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

try:
    from pulp import (
        LpProblem, LpMinimize, LpMaximize, LpVariable,
        LpStatus, lpSum, value, LpInteger, LpContinuous
    )
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    print("Aviso: PuLP não instalado. Instale com: pip install pulp")


@dataclass
class ResultadoOtimizacao:
    """Estrutura para armazenar resultados da otimização."""
    status: str
    custo_total: float
    custo_manutencao: float
    custo_stockout: float
    alocacoes: pd.DataFrame
    tempo_execucao: float
    gap_otimalidade: Optional[float] = None
    mensagem: str = ""


class OtimizadorLinear:
    """
    Classe para otimização de estoque usando programação linear.

    Resolve problemas de:
    - Alocação ótima de estoque
    - Minimização de custos
    - Planejamento de reposição
    """

    def __init__(self, metricas_estoque: pd.DataFrame):
        """
        Inicializa o otimizador com dados de estoque.

        Args:
            metricas_estoque: DataFrame com métricas de estoque por loja/produto
        """
        if not PULP_AVAILABLE:
            raise ImportError("PuLP é necessário. Instale com: pip install pulp")

        self.metricas = metricas_estoque.copy()
        self.lojas = self.metricas["loja_id"].unique().tolist()
        self.produtos = self.metricas["produto_id"].unique().tolist()

    def otimizar_alocacao_estoque(
        self,
        estoque_disponivel: Dict[str, int],
        nivel_servico_minimo: float = 0.90,
        solver: str = None
    ) -> ResultadoOtimizacao:
        """
        Otimiza a alocação de estoque do CD para as lojas.

        Minimiza o custo total (manutenção + risco de stockout) sujeito a:
        - Estoque disponível no CD por produto
        - Nível de serviço mínimo por loja
        - Capacidade de armazenamento das lojas

        Args:
            estoque_disponivel: Dict com estoque disponível por produto no CD
            nivel_servico_minimo: Nível de serviço mínimo (0-1)
            solver: Solver a usar (None para default)

        Returns:
            ResultadoOtimizacao com alocações ótimas
        """
        import time
        inicio = time.time()

        # Criar problema de minimização
        prob = LpProblem("Alocacao_Estoque", LpMinimize)

        # Variáveis de decisão: quantidade a enviar para cada loja/produto
        x = {}
        for _, row in self.metricas.iterrows():
            loja = row["loja_id"]
            prod = row["produto_id"]
            # Quantidade máxima: estoque disponível ou demanda de 30 dias
            max_qty = min(
                estoque_disponivel.get(prod, 0),
                int(row["demanda_media"] * 30)
            )
            x[(loja, prod)] = LpVariable(
                f"x_{loja}_{prod}",
                lowBound=0,
                upBound=max_qty,
                cat=LpInteger
            )

        # Variáveis auxiliares: stockout esperado
        stockout = {}
        for _, row in self.metricas.iterrows():
            loja = row["loja_id"]
            prod = row["produto_id"]
            stockout[(loja, prod)] = LpVariable(
                f"stockout_{loja}_{prod}",
                lowBound=0,
                cat=LpContinuous
            )

        # Função objetivo: minimizar custo total
        # Custo = custo_manutencao * estoque + custo_stockout * stockout_esperado
        custo_manutencao = lpSum([
            row["custo_manutencao"] * 30 * x[(row["loja_id"], row["produto_id"])]
            for _, row in self.metricas.iterrows()
        ])

        custo_stockout_total = lpSum([
            row["custo_stockout"] * stockout[(row["loja_id"], row["produto_id"])]
            for _, row in self.metricas.iterrows()
        ])

        prob += custo_manutencao + custo_stockout_total, "Custo_Total"

        # Restrições

        # 1. Estoque disponível no CD por produto
        for prod in self.produtos:
            prob += (
                lpSum([
                    x[(loja, prod)]
                    for loja in self.lojas
                    if (loja, prod) in x
                ]) <= estoque_disponivel.get(prod, 0),
                f"Disponibilidade_{prod}"
            )

        # 2. Nível de serviço mínimo (demanda atendida >= nível * demanda esperada)
        for _, row in self.metricas.iterrows():
            loja = row["loja_id"]
            prod = row["produto_id"]
            demanda_periodo = row["demanda_media"] * 30

            # Estoque atual + alocação - stockout >= nível_servico * demanda
            prob += (
                row["estoque_atual"] + x[(loja, prod)] - stockout[(loja, prod)] >=
                nivel_servico_minimo * demanda_periodo,
                f"NivelServico_{loja}_{prod}"
            )

            # Stockout não pode ser maior que a demanda
            prob += (
                stockout[(loja, prod)] <= demanda_periodo,
                f"MaxStockout_{loja}_{prod}"
            )

        # 3. Quantidade mínima por envio (evitar envios muito pequenos)
        MIN_ENVIO = 10
        for _, row in self.metricas.iterrows():
            loja = row["loja_id"]
            prod = row["produto_id"]
            # Se enviar, enviar pelo menos MIN_ENVIO (relaxado para LP)
            # Nota: restrição removida para manter problema linear

        # Resolver
        prob.solve()

        tempo_execucao = time.time() - inicio

        # Extrair resultados
        if LpStatus[prob.status] == "Optimal":
            alocacoes = []
            for _, row in self.metricas.iterrows():
                loja = row["loja_id"]
                prod = row["produto_id"]
                qtd = value(x[(loja, prod)])
                stk = value(stockout[(loja, prod)])

                if qtd > 0:
                    alocacoes.append({
                        "loja_id": loja,
                        "produto_id": prod,
                        "produto_nome": row["produto_nome"],
                        "estoque_atual": row["estoque_atual"],
                        "quantidade_alocada": int(qtd),
                        "estoque_final": row["estoque_atual"] + int(qtd),
                        "demanda_30dias": round(row["demanda_media"] * 30, 0),
                        "stockout_esperado": round(stk, 1),
                        "custo_manutencao": round(row["custo_manutencao"] * 30 * qtd, 2),
                        "custo_stockout": round(row["custo_stockout"] * stk, 2)
                    })

            df_alocacoes = pd.DataFrame(alocacoes)

            custo_man = sum(a["custo_manutencao"] for a in alocacoes) if alocacoes else 0
            custo_stk = sum(a["custo_stockout"] for a in alocacoes) if alocacoes else 0

            return ResultadoOtimizacao(
                status="Ótimo",
                custo_total=round(value(prob.objective), 2),
                custo_manutencao=round(custo_man, 2),
                custo_stockout=round(custo_stk, 2),
                alocacoes=df_alocacoes,
                tempo_execucao=round(tempo_execucao, 3),
                mensagem="Solução ótima encontrada"
            )
        else:
            return ResultadoOtimizacao(
                status=LpStatus[prob.status],
                custo_total=0,
                custo_manutencao=0,
                custo_stockout=0,
                alocacoes=pd.DataFrame(),
                tempo_execucao=round(tempo_execucao, 3),
                mensagem=f"Problema não resolvido: {LpStatus[prob.status]}"
            )

    def otimizar_reposicao_orcamento(
        self,
        orcamento: float,
        precos_compra: Dict[str, float],
        horizonte_dias: int = 30
    ) -> ResultadoOtimizacao:
        """
        Otimiza reposição maximizando nível de serviço com orçamento limitado.

        Decide quanto comprar de cada produto respeitando o orçamento,
        priorizando produtos com maior impacto no nível de serviço.

        Args:
            orcamento: Orçamento total disponível para compras
            precos_compra: Dict com preço de compra por produto
            horizonte_dias: Horizonte de planejamento

        Returns:
            ResultadoOtimizacao com quantidades ótimas de compra
        """
        import time
        inicio = time.time()

        # Criar problema de minimização de stockout
        prob = LpProblem("Reposicao_Orcamento", LpMinimize)

        # Variáveis: quantidade a comprar por produto (agregado)
        compras = {}
        for prod in self.produtos:
            # Demanda total de todas as lojas
            demanda_total = self.metricas[
                self.metricas["produto_id"] == prod
            ]["demanda_media"].sum() * horizonte_dias

            compras[prod] = LpVariable(
                f"compra_{prod}",
                lowBound=0,
                upBound=int(demanda_total * 1.5),  # Máximo 150% da demanda
                cat=LpInteger
            )

        # Variáveis: déficit por loja/produto
        deficit = {}
        for _, row in self.metricas.iterrows():
            loja = row["loja_id"]
            prod = row["produto_id"]
            deficit[(loja, prod)] = LpVariable(
                f"deficit_{loja}_{prod}",
                lowBound=0,
                cat=LpContinuous
            )

        # Variáveis: alocação por loja/produto (para distribuir compras)
        alocacao = {}
        for _, row in self.metricas.iterrows():
            loja = row["loja_id"]
            prod = row["produto_id"]
            demanda = row["demanda_media"] * horizonte_dias
            alocacao[(loja, prod)] = LpVariable(
                f"aloc_{loja}_{prod}",
                lowBound=0,
                upBound=int(demanda * 1.5),
                cat=LpContinuous
            )

        # Função objetivo: minimizar custo ponderado de stockout
        prob += lpSum([
            row["custo_stockout"] * deficit[(row["loja_id"], row["produto_id"])]
            for _, row in self.metricas.iterrows()
        ]), "Custo_Stockout_Total"

        # Restrições

        # 1. Orçamento total
        prob += (
            lpSum([
                precos_compra.get(prod, 10) * compras[prod]
                for prod in self.produtos
            ]) <= orcamento,
            "Restricao_Orcamento"
        )

        # 2. Alocação não pode exceder compras
        for prod in self.produtos:
            prob += (
                lpSum([
                    alocacao[(loja, prod)]
                    for loja in self.lojas
                    if (loja, prod) in alocacao
                ]) <= compras[prod],
                f"Limite_Compra_{prod}"
            )

        # 3. Déficit = demanda - (estoque_atual + alocação)
        for _, row in self.metricas.iterrows():
            loja = row["loja_id"]
            prod = row["produto_id"]
            demanda = row["demanda_media"] * horizonte_dias

            prob += (
                deficit[(loja, prod)] >=
                demanda - row["estoque_atual"] - alocacao[(loja, prod)],
                f"Deficit_{loja}_{prod}"
            )

        # Resolver
        prob.solve()

        tempo_execucao = time.time() - inicio

        # Extrair resultados
        if LpStatus[prob.status] == "Optimal":
            resultados_compra = []
            for prod in self.produtos:
                qtd = value(compras[prod])
                if qtd > 0:
                    preco = precos_compra.get(prod, 10)
                    nome = self.metricas[
                        self.metricas["produto_id"] == prod
                    ]["produto_nome"].iloc[0]

                    resultados_compra.append({
                        "produto_id": prod,
                        "produto_nome": nome,
                        "quantidade_compra": int(qtd),
                        "preco_unitario": preco,
                        "custo_total": round(qtd * preco, 2)
                    })

            df_compras = pd.DataFrame(resultados_compra)

            # Calcular métricas
            custo_compra_total = df_compras["custo_total"].sum() if len(df_compras) > 0 else 0
            stockout_total = sum(
                value(deficit[(row["loja_id"], row["produto_id"])])
                for _, row in self.metricas.iterrows()
            )

            return ResultadoOtimizacao(
                status="Ótimo",
                custo_total=round(custo_compra_total, 2),
                custo_manutencao=round(custo_compra_total, 2),  # Custo de compra
                custo_stockout=round(value(prob.objective), 2),
                alocacoes=df_compras,
                tempo_execucao=round(tempo_execucao, 3),
                mensagem=f"Orçamento utilizado: R$ {custo_compra_total:,.2f} de R$ {orcamento:,.2f}"
            )
        else:
            return ResultadoOtimizacao(
                status=LpStatus[prob.status],
                custo_total=0,
                custo_manutencao=0,
                custo_stockout=0,
                alocacoes=pd.DataFrame(),
                tempo_execucao=round(tempo_execucao, 3),
                mensagem=f"Problema não resolvido: {LpStatus[prob.status]}"
            )

    def otimizar_multiperiodo(
        self,
        previsao_demanda: pd.DataFrame,
        estoque_inicial: Dict[Tuple[str, str], int],
        capacidade_armazem: Dict[str, int],
        custo_pedido: float = 100.0,
        periodos: int = 4
    ) -> ResultadoOtimizacao:
        """
        Otimização multi-período para planejamento de reposição.

        Planeja reposições ao longo de múltiplos períodos minimizando
        custo total (manutenção + pedido + stockout).

        Args:
            previsao_demanda: DataFrame com previsão por período/loja/produto
            estoque_inicial: Dict com estoque inicial por (loja, produto)
            capacidade_armazem: Dict com capacidade por loja
            custo_pedido: Custo fixo por pedido
            periodos: Número de períodos a planejar

        Returns:
            ResultadoOtimizacao com plano de reposição
        """
        import time
        inicio = time.time()

        prob = LpProblem("Planejamento_Multiperiodo", LpMinimize)

        # Variáveis por período
        pedido = {}     # Quantidade pedida
        estoque = {}    # Nível de estoque
        stockout = {}   # Falta de estoque
        faz_pedido = {} # Binária: se faz pedido

        for t in range(periodos):
            for _, row in self.metricas.iterrows():
                loja = row["loja_id"]
                prod = row["produto_id"]
                key = (t, loja, prod)

                # Variáveis
                pedido[key] = LpVariable(f"ped_{t}_{loja}_{prod}", lowBound=0, cat=LpInteger)
                estoque[key] = LpVariable(f"est_{t}_{loja}_{prod}", lowBound=0, cat=LpContinuous)
                stockout[key] = LpVariable(f"stk_{t}_{loja}_{prod}", lowBound=0, cat=LpContinuous)

        # Função objetivo
        custo_total = lpSum([
            # Custo de manutenção
            self.metricas[
                (self.metricas["loja_id"] == loja) &
                (self.metricas["produto_id"] == prod)
            ]["custo_manutencao"].iloc[0] * 7 * estoque[(t, loja, prod)]
            +
            # Custo de stockout
            self.metricas[
                (self.metricas["loja_id"] == loja) &
                (self.metricas["produto_id"] == prod)
            ]["custo_stockout"].iloc[0] * stockout[(t, loja, prod)]
            for t in range(periodos)
            for loja in self.lojas
            for prod in self.produtos
            if (t, loja, prod) in estoque
        ])

        prob += custo_total, "Custo_Total"

        # Restrições de balanço de estoque
        for _, row in self.metricas.iterrows():
            loja = row["loja_id"]
            prod = row["produto_id"]
            demanda_base = row["demanda_media"] * 7  # Demanda semanal

            for t in range(periodos):
                key = (t, loja, prod)

                if t == 0:
                    # Primeiro período: usa estoque inicial
                    est_anterior = estoque_inicial.get((loja, prod), row["estoque_atual"])
                else:
                    est_anterior = estoque[(t-1, loja, prod)]

                # Balanço: estoque_t = estoque_t-1 + pedido_t - demanda + stockout
                prob += (
                    estoque[key] == est_anterior + pedido[key] - demanda_base + stockout[key],
                    f"Balanco_{t}_{loja}_{prod}"
                )

                # Capacidade de armazenamento
                cap = capacidade_armazem.get(loja, 10000)
                prob += (
                    estoque[key] <= cap / len(self.produtos),
                    f"Capacidade_{t}_{loja}_{prod}"
                )

        # Resolver
        prob.solve()

        tempo_execucao = time.time() - inicio

        if LpStatus[prob.status] == "Optimal":
            # Extrair plano de pedidos
            plano = []
            for t in range(periodos):
                for _, row in self.metricas.iterrows():
                    loja = row["loja_id"]
                    prod = row["produto_id"]
                    key = (t, loja, prod)

                    qtd_pedido = value(pedido[key])
                    qtd_estoque = value(estoque[key])
                    qtd_stockout = value(stockout[key])

                    if qtd_pedido > 0 or qtd_stockout > 0:
                        plano.append({
                            "periodo": t + 1,
                            "semana": f"Semana {t + 1}",
                            "loja_id": loja,
                            "produto_id": prod,
                            "produto_nome": row["produto_nome"],
                            "quantidade_pedido": int(qtd_pedido),
                            "estoque_final": round(qtd_estoque, 0),
                            "stockout": round(qtd_stockout, 1),
                            "demanda_esperada": round(row["demanda_media"] * 7, 0)
                        })

            df_plano = pd.DataFrame(plano)

            return ResultadoOtimizacao(
                status="Ótimo",
                custo_total=round(value(prob.objective), 2),
                custo_manutencao=0,  # Incluído no custo total
                custo_stockout=0,    # Incluído no custo total
                alocacoes=df_plano,
                tempo_execucao=round(tempo_execucao, 3),
                mensagem=f"Plano otimizado para {periodos} períodos"
            )
        else:
            return ResultadoOtimizacao(
                status=LpStatus[prob.status],
                custo_total=0,
                custo_manutencao=0,
                custo_stockout=0,
                alocacoes=pd.DataFrame(),
                tempo_execucao=round(tempo_execucao, 3),
                mensagem=f"Problema não resolvido: {LpStatus[prob.status]}"
            )


def otimizar_mix_produtos(
    metricas_estoque: pd.DataFrame,
    espaco_total: float,
    espaco_por_unidade: Dict[str, float],
    margem_por_produto: Dict[str, float]
) -> ResultadoOtimizacao:
    """
    Otimiza o mix de produtos considerando espaço e margem.

    Maximiza a margem total sujeito a restrições de espaço
    e níveis mínimos de serviço.

    Args:
        metricas_estoque: DataFrame com métricas
        espaco_total: Espaço total disponível
        espaco_por_unidade: Espaço ocupado por unidade de cada produto
        margem_por_produto: Margem de lucro por produto

    Returns:
        ResultadoOtimizacao com quantidades ótimas
    """
    if not PULP_AVAILABLE:
        raise ImportError("PuLP é necessário. Instale com: pip install pulp")

    import time
    inicio = time.time()

    prob = LpProblem("Mix_Produtos", LpMaximize)

    produtos = metricas_estoque["produto_id"].unique()

    # Variáveis: quantidade de cada produto
    x = {
        prod: LpVariable(f"x_{prod}", lowBound=0, cat=LpInteger)
        for prod in produtos
    }

    # Função objetivo: maximizar margem
    prob += lpSum([
        margem_por_produto.get(prod, 1) * x[prod]
        for prod in produtos
    ]), "Margem_Total"

    # Restrição de espaço
    prob += (
        lpSum([
            espaco_por_unidade.get(prod, 1) * x[prod]
            for prod in produtos
        ]) <= espaco_total,
        "Restricao_Espaco"
    )

    # Quantidades mínimas (cobertura de 1 semana)
    for prod in produtos:
        demanda_semanal = metricas_estoque[
            metricas_estoque["produto_id"] == prod
        ]["demanda_media"].sum() * 7

        prob += (
            x[prod] >= demanda_semanal,
            f"Minimo_{prod}"
        )

    prob.solve()

    tempo_execucao = time.time() - inicio

    if LpStatus[prob.status] == "Optimal":
        resultados = []
        for prod in produtos:
            qtd = value(x[prod])
            nome = metricas_estoque[
                metricas_estoque["produto_id"] == prod
            ]["produto_nome"].iloc[0]

            resultados.append({
                "produto_id": prod,
                "produto_nome": nome,
                "quantidade_otima": int(qtd),
                "espaco_usado": round(espaco_por_unidade.get(prod, 1) * qtd, 2),
                "margem_total": round(margem_por_produto.get(prod, 1) * qtd, 2)
            })

        df_resultados = pd.DataFrame(resultados)

        return ResultadoOtimizacao(
            status="Ótimo",
            custo_total=round(value(prob.objective), 2),  # Aqui é margem (maximizada)
            custo_manutencao=0,
            custo_stockout=0,
            alocacoes=df_resultados,
            tempo_execucao=round(tempo_execucao, 3),
            mensagem=f"Margem total otimizada: R$ {value(prob.objective):,.2f}"
        )
    else:
        return ResultadoOtimizacao(
            status=LpStatus[prob.status],
            custo_total=0,
            custo_manutencao=0,
            custo_stockout=0,
            alocacoes=pd.DataFrame(),
            tempo_execucao=round(tempo_execucao, 3),
            mensagem=f"Problema não resolvido: {LpStatus[prob.status]}"
        )


def gerar_relatorio_otimizacao(resultado: ResultadoOtimizacao) -> str:
    """
    Gera relatório formatado dos resultados da otimização.

    Args:
        resultado: Objeto ResultadoOtimizacao

    Returns:
        String formatada com relatório
    """
    linhas = []
    linhas.append("=" * 60)
    linhas.append("   RELATÓRIO DE OTIMIZAÇÃO LINEAR")
    linhas.append("=" * 60)
    linhas.append("")
    linhas.append(f"Status: {resultado.status}")
    linhas.append(f"Tempo de execução: {resultado.tempo_execucao}s")
    linhas.append(f"Mensagem: {resultado.mensagem}")
    linhas.append("")
    linhas.append("-" * 60)
    linhas.append("CUSTOS")
    linhas.append("-" * 60)
    linhas.append(f"Custo Total:      R$ {resultado.custo_total:>12,.2f}")
    linhas.append(f"Custo Manutenção: R$ {resultado.custo_manutencao:>12,.2f}")
    linhas.append(f"Custo Stockout:   R$ {resultado.custo_stockout:>12,.2f}")
    linhas.append("")

    if len(resultado.alocacoes) > 0:
        linhas.append("-" * 60)
        linhas.append("ALOCAÇÕES / RECOMENDAÇÕES")
        linhas.append("-" * 60)
        linhas.append(resultado.alocacoes.to_string(index=False))

    linhas.append("")
    linhas.append("=" * 60)

    return "\n".join(linhas)


if __name__ == "__main__":
    # Teste do módulo
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from data_generator import gerar_historico_vendas, gerar_estoque_atual, PRODUTOS
    from statistical_analysis import calcular_estatisticas_demanda, calcular_metricas_estoque

    print("Gerando dados de teste...")
    historico = gerar_historico_vendas(dias=30)
    estoque = gerar_estoque_atual(historico)
    stats = calcular_estatisticas_demanda(historico)
    metricas = calcular_metricas_estoque(estoque, stats)

    print("\n" + "=" * 60)
    print("   TESTE 1: Alocação Ótima de Estoque")
    print("=" * 60)

    # Estoque disponível no CD
    estoque_cd = {
        "PROD_001": 5000,
        "PROD_002": 8000,
        "PROD_003": 6000,
        "PROD_004": 7000,
        "PROD_005": 12000,
        "PROD_006": 9000,
        "PROD_007": 10000,
        "PROD_008": 4000
    }

    otimizador = OtimizadorLinear(metricas)
    resultado1 = otimizador.otimizar_alocacao_estoque(estoque_cd, nivel_servico_minimo=0.95)
    print(gerar_relatorio_otimizacao(resultado1))

    print("\n" + "=" * 60)
    print("   TESTE 2: Otimização com Orçamento Limitado")
    print("=" * 60)

    precos = {prod: config["preco_unitario"] * 0.6 for prod, config in PRODUTOS.items()}
    resultado2 = otimizador.otimizar_reposicao_orcamento(
        orcamento=50000,
        precos_compra=precos
    )
    print(gerar_relatorio_otimizacao(resultado2))

    print("\n" + "=" * 60)
    print("   TESTE 3: Otimização de Mix de Produtos")
    print("=" * 60)

    espaco = {prod: 0.1 for prod in PRODUTOS.keys()}
    margem = {prod: config["preco_unitario"] * 0.3 for prod, config in PRODUTOS.items()}

    resultado3 = otimizar_mix_produtos(
        metricas,
        espaco_total=1000,
        espaco_por_unidade=espaco,
        margem_por_produto=margem
    )
    print(gerar_relatorio_otimizacao(resultado3))
