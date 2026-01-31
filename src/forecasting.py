"""
Módulo de Previsão de Demanda
=============================
Implementa métodos de previsão usando Exponential Smoothing.

Métodos disponíveis:
1. Simple Exponential Smoothing (SES) - séries estacionárias
2. Holt's Linear Trend - séries com tendência
3. Holt-Winters - séries com tendência e sazonalidade

Biblioteca: statsmodels
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

try:
    from statsmodels.tsa.holtwinters import (
        ExponentialSmoothing,
        SimpleExpSmoothing,
        Holt
    )
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Aviso: statsmodels não instalado. Instale com: pip install statsmodels")


@dataclass
class ResultadoPrevisao:
    """Estrutura para armazenar resultados da previsão."""
    produto_id: str
    loja_id: Optional[str]
    metodo: str
    previsoes: pd.DataFrame
    metricas: Dict[str, float]
    parametros: Dict[str, any]
    intervalo_confianca: Optional[pd.DataFrame] = None


class PrevisaoDemanda:
    """
    Classe para previsão de demanda usando Exponential Smoothing.

    Suporta:
    - Simple Exponential Smoothing (SES)
    - Holt's Linear Trend Method
    - Holt-Winters (aditivo e multiplicativo)
    """

    def __init__(self, historico_vendas: pd.DataFrame):
        """
        Inicializa o previsor com histórico de vendas.

        Args:
            historico_vendas: DataFrame com colunas [data, loja_id, produto_id, quantidade_vendida]
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels é necessário. Instale com: pip install statsmodels")

        self.historico = historico_vendas.copy()
        self.historico["data"] = pd.to_datetime(self.historico["data"])

    def preparar_serie_temporal(
        self,
        produto_id: str,
        loja_id: Optional[str] = None,
        frequencia: str = "D"
    ) -> pd.Series:
        """
        Prepara série temporal para um produto/loja.

        Args:
            produto_id: ID do produto
            loja_id: ID da loja (None para agregar todas)
            frequencia: Frequência da série ('D'=diário, 'W'=semanal)

        Returns:
            Series com índice temporal
        """
        df = self.historico[self.historico["produto_id"] == produto_id].copy()

        if loja_id:
            df = df[df["loja_id"] == loja_id]

        # Agregar por data
        serie = df.groupby("data")["quantidade_vendida"].sum()

        # Garantir frequência contínua
        idx = pd.date_range(start=serie.index.min(), end=serie.index.max(), freq=frequencia)
        serie = serie.reindex(idx, fill_value=0)

        # Tratar zeros (substituir por média móvel para evitar problemas)
        if (serie == 0).any():
            serie = serie.replace(0, serie.rolling(3, min_periods=1).mean())

        return serie

    def simple_exponential_smoothing(
        self,
        produto_id: str,
        loja_id: Optional[str] = None,
        horizonte: int = 7,
        alpha: Optional[float] = None
    ) -> ResultadoPrevisao:
        """
        Previsão usando Simple Exponential Smoothing.

        Ideal para séries sem tendência ou sazonalidade clara.

        Args:
            produto_id: ID do produto
            loja_id: ID da loja
            horizonte: Dias a prever
            alpha: Parâmetro de suavização (None para otimizar)

        Returns:
            ResultadoPrevisao com previsões e métricas
        """
        serie = self.preparar_serie_temporal(produto_id, loja_id)

        # Ajustar modelo
        if alpha:
            modelo = SimpleExpSmoothing(serie, initialization_method="estimated")
            resultado = modelo.fit(smoothing_level=alpha, optimized=False)
        else:
            modelo = SimpleExpSmoothing(serie, initialization_method="estimated")
            resultado = modelo.fit(optimized=True)

        # Gerar previsões
        previsoes = resultado.forecast(horizonte)

        # Calcular métricas no conjunto de treino
        fitted = resultado.fittedvalues
        metricas = self._calcular_metricas(serie, fitted)

        # Criar DataFrame de previsões
        datas_futuras = pd.date_range(
            start=serie.index[-1] + timedelta(days=1),
            periods=horizonte,
            freq="D"
        )

        df_previsoes = pd.DataFrame({
            "data": datas_futuras,
            "previsao": previsoes.values,
            "limite_inferior": previsoes.values * 0.8,  # Aproximação simples
            "limite_superior": previsoes.values * 1.2
        })

        return ResultadoPrevisao(
            produto_id=produto_id,
            loja_id=loja_id,
            metodo="Simple Exponential Smoothing",
            previsoes=df_previsoes,
            metricas=metricas,
            parametros={"alpha": resultado.params.get("smoothing_level", alpha)}
        )

    def holt_linear(
        self,
        produto_id: str,
        loja_id: Optional[str] = None,
        horizonte: int = 7,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        damped: bool = False
    ) -> ResultadoPrevisao:
        """
        Previsão usando Holt's Linear Trend Method.

        Ideal para séries com tendência mas sem sazonalidade.

        Args:
            produto_id: ID do produto
            loja_id: ID da loja
            horizonte: Dias a prever
            alpha: Parâmetro de suavização do nível
            beta: Parâmetro de suavização da tendência
            damped: Se True, usa tendência amortecida

        Returns:
            ResultadoPrevisao com previsões e métricas
        """
        serie = self.preparar_serie_temporal(produto_id, loja_id)

        # Ajustar modelo
        modelo = Holt(serie, damped_trend=damped, initialization_method="estimated")

        if alpha and beta:
            resultado = modelo.fit(
                smoothing_level=alpha,
                smoothing_trend=beta,
                optimized=False
            )
        else:
            resultado = modelo.fit(optimized=True)

        # Gerar previsões
        previsoes = resultado.forecast(horizonte)

        # Calcular métricas
        fitted = resultado.fittedvalues
        metricas = self._calcular_metricas(serie, fitted)

        # Criar DataFrame de previsões
        datas_futuras = pd.date_range(
            start=serie.index[-1] + timedelta(days=1),
            periods=horizonte,
            freq="D"
        )

        # Intervalo de confiança aproximado
        residuos_std = (serie - fitted).std()
        df_previsoes = pd.DataFrame({
            "data": datas_futuras,
            "previsao": previsoes.values,
            "limite_inferior": previsoes.values - 1.96 * residuos_std,
            "limite_superior": previsoes.values + 1.96 * residuos_std
        })

        return ResultadoPrevisao(
            produto_id=produto_id,
            loja_id=loja_id,
            metodo="Holt Linear Trend" + (" (Damped)" if damped else ""),
            previsoes=df_previsoes,
            metricas=metricas,
            parametros={
                "alpha": resultado.params.get("smoothing_level"),
                "beta": resultado.params.get("smoothing_trend"),
                "damped": damped
            }
        )

    def holt_winters(
        self,
        produto_id: str,
        loja_id: Optional[str] = None,
        horizonte: int = 7,
        periodo_sazonal: int = 7,
        tipo_sazonal: str = "add",
        tipo_tendencia: str = "add",
        damped: bool = False
    ) -> ResultadoPrevisao:
        """
        Previsão usando Holt-Winters (Triple Exponential Smoothing).

        Ideal para séries com tendência E sazonalidade.

        Args:
            produto_id: ID do produto
            loja_id: ID da loja
            horizonte: Dias a prever
            periodo_sazonal: Período da sazonalidade (7 para semanal)
            tipo_sazonal: 'add' (aditivo) ou 'mul' (multiplicativo)
            tipo_tendencia: 'add' (aditivo) ou 'mul' (multiplicativo)
            damped: Se True, usa tendência amortecida

        Returns:
            ResultadoPrevisao com previsões e métricas
        """
        serie = self.preparar_serie_temporal(produto_id, loja_id)

        # Verificar se há dados suficientes para sazonalidade
        if len(serie) < 2 * periodo_sazonal:
            warnings.warn(
                f"Dados insuficientes para Holt-Winters com período {periodo_sazonal}. "
                f"Usando Holt Linear."
            )
            return self.holt_linear(produto_id, loja_id, horizonte, damped=damped)

        try:
            # Ajustar modelo Holt-Winters
            modelo = ExponentialSmoothing(
                serie,
                trend=tipo_tendencia,
                seasonal=tipo_sazonal,
                seasonal_periods=periodo_sazonal,
                damped_trend=damped,
                initialization_method="estimated"
            )

            resultado = modelo.fit(optimized=True)

            # Gerar previsões
            previsoes = resultado.forecast(horizonte)

            # Calcular métricas
            fitted = resultado.fittedvalues
            metricas = self._calcular_metricas(serie, fitted)

            # Criar DataFrame de previsões com intervalo de confiança
            datas_futuras = pd.date_range(
                start=serie.index[-1] + timedelta(days=1),
                periods=horizonte,
                freq="D"
            )

            # Intervalo de confiança aproximado
            residuos_std = (serie - fitted).std()
            df_previsoes = pd.DataFrame({
                "data": datas_futuras,
                "previsao": previsoes.values,
                "limite_inferior": previsoes.values - 1.96 * residuos_std,
                "limite_superior": previsoes.values + 1.96 * residuos_std
            })

            # Garantir limites não negativos
            df_previsoes["limite_inferior"] = df_previsoes["limite_inferior"].clip(lower=0)
            df_previsoes["previsao"] = df_previsoes["previsao"].clip(lower=0)

            return ResultadoPrevisao(
                produto_id=produto_id,
                loja_id=loja_id,
                metodo=f"Holt-Winters ({tipo_sazonal}/{tipo_tendencia})",
                previsoes=df_previsoes,
                metricas=metricas,
                parametros={
                    "alpha": resultado.params.get("smoothing_level"),
                    "beta": resultado.params.get("smoothing_trend"),
                    "gamma": resultado.params.get("smoothing_seasonal"),
                    "periodo_sazonal": periodo_sazonal,
                    "tipo_sazonal": tipo_sazonal,
                    "tipo_tendencia": tipo_tendencia,
                    "damped": damped
                }
            )

        except Exception as e:
            warnings.warn(f"Erro no Holt-Winters: {e}. Usando Holt Linear.")
            return self.holt_linear(produto_id, loja_id, horizonte, damped=damped)

    def auto_selecionar_metodo(
        self,
        produto_id: str,
        loja_id: Optional[str] = None,
        horizonte: int = 7
    ) -> ResultadoPrevisao:
        """
        Seleciona automaticamente o melhor método de previsão.

        Testa múltiplos métodos e escolhe o com menor erro (MAPE).

        Args:
            produto_id: ID do produto
            loja_id: ID da loja
            horizonte: Dias a prever

        Returns:
            ResultadoPrevisao do melhor método
        """
        serie = self.preparar_serie_temporal(produto_id, loja_id)

        candidatos = []

        # Testar SES
        try:
            resultado_ses = self.simple_exponential_smoothing(
                produto_id, loja_id, horizonte
            )
            candidatos.append(resultado_ses)
        except Exception:
            pass

        # Testar Holt Linear
        try:
            resultado_holt = self.holt_linear(produto_id, loja_id, horizonte)
            candidatos.append(resultado_holt)
        except Exception:
            pass

        # Testar Holt Linear Damped
        try:
            resultado_holt_d = self.holt_linear(
                produto_id, loja_id, horizonte, damped=True
            )
            candidatos.append(resultado_holt_d)
        except Exception:
            pass

        # Testar Holt-Winters (se houver dados suficientes)
        if len(serie) >= 14:
            try:
                resultado_hw = self.holt_winters(
                    produto_id, loja_id, horizonte,
                    periodo_sazonal=7, tipo_sazonal="add"
                )
                candidatos.append(resultado_hw)
            except Exception:
                pass

        # Selecionar melhor baseado no MAPE
        if candidatos:
            melhor = min(candidatos, key=lambda x: x.metricas.get("mape", float("inf")))
            return melhor
        else:
            raise ValueError("Nenhum método de previsão funcionou para este produto.")

    def prever_todos_produtos(
        self,
        horizonte: int = 7,
        metodo: str = "auto"
    ) -> Dict[str, ResultadoPrevisao]:
        """
        Gera previsões para todos os produtos (agregado).

        Args:
            horizonte: Dias a prever
            metodo: 'auto', 'ses', 'holt', 'holt_winters'

        Returns:
            Dict com previsões por produto
        """
        produtos = self.historico["produto_id"].unique()
        resultados = {}

        for produto_id in produtos:
            try:
                if metodo == "auto":
                    resultado = self.auto_selecionar_metodo(produto_id, None, horizonte)
                elif metodo == "ses":
                    resultado = self.simple_exponential_smoothing(produto_id, None, horizonte)
                elif metodo == "holt":
                    resultado = self.holt_linear(produto_id, None, horizonte)
                elif metodo == "holt_winters":
                    resultado = self.holt_winters(produto_id, None, horizonte)
                else:
                    resultado = self.auto_selecionar_metodo(produto_id, None, horizonte)

                resultados[produto_id] = resultado
            except Exception as e:
                warnings.warn(f"Erro ao prever {produto_id}: {e}")

        return resultados

    def decompor_serie(
        self,
        produto_id: str,
        loja_id: Optional[str] = None,
        periodo: int = 7
    ) -> Dict:
        """
        Decompõe série temporal em tendência, sazonalidade e resíduo.

        Args:
            produto_id: ID do produto
            loja_id: ID da loja
            periodo: Período sazonal

        Returns:
            Dict com componentes da decomposição
        """
        serie = self.preparar_serie_temporal(produto_id, loja_id)

        if len(serie) < 2 * periodo:
            return {
                "erro": "Dados insuficientes para decomposição",
                "serie": serie
            }

        decomposicao = seasonal_decompose(serie, model="additive", period=periodo)

        return {
            "tendencia": decomposicao.trend,
            "sazonalidade": decomposicao.seasonal,
            "residuo": decomposicao.resid,
            "serie_original": serie
        }

    def _calcular_metricas(
        self,
        real: pd.Series,
        previsto: pd.Series
    ) -> Dict[str, float]:
        """
        Calcula métricas de erro da previsão.

        Args:
            real: Valores reais
            previsto: Valores previstos

        Returns:
            Dict com MAE, RMSE, MAPE
        """
        # Alinhar séries
        real = real.dropna()
        previsto = previsto.reindex(real.index).dropna()

        if len(real) == 0 or len(previsto) == 0:
            return {"mae": 0, "rmse": 0, "mape": 0}

        # Usar interseção
        idx = real.index.intersection(previsto.index)
        real = real[idx]
        previsto = previsto[idx]

        # MAE
        mae = np.abs(real - previsto).mean()

        # RMSE
        rmse = np.sqrt(((real - previsto) ** 2).mean())

        # MAPE (evitar divisão por zero)
        mape_valores = np.abs((real - previsto) / real.replace(0, np.nan)) * 100
        mape = mape_valores.mean()

        return {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 2) if not np.isnan(mape) else 0
        }


def gerar_relatorio_previsao(
    resultado: ResultadoPrevisao,
    nome_produto: str = None
) -> str:
    """
    Gera relatório formatado da previsão.

    Args:
        resultado: ResultadoPrevisao
        nome_produto: Nome do produto (opcional)

    Returns:
        String formatada
    """
    linhas = []
    linhas.append("=" * 60)
    linhas.append("   RELATÓRIO DE PREVISÃO DE DEMANDA")
    linhas.append("=" * 60)
    linhas.append("")
    linhas.append(f"Produto: {nome_produto or resultado.produto_id}")
    if resultado.loja_id:
        linhas.append(f"Loja: {resultado.loja_id}")
    else:
        linhas.append("Loja: TODAS (agregado)")
    linhas.append(f"Método: {resultado.metodo}")
    linhas.append("")
    linhas.append("-" * 60)
    linhas.append("MÉTRICAS DE QUALIDADE")
    linhas.append("-" * 60)
    linhas.append(f"   MAE (Erro Absoluto Médio):  {resultado.metricas['mae']:>10.2f}")
    linhas.append(f"   RMSE (Raiz do Erro Quadr.): {resultado.metricas['rmse']:>10.2f}")
    linhas.append(f"   MAPE (Erro % Médio):        {resultado.metricas['mape']:>10.2f}%")
    linhas.append("")
    linhas.append("-" * 60)
    linhas.append("PREVISÕES")
    linhas.append("-" * 60)

    for _, row in resultado.previsoes.iterrows():
        data_str = row["data"].strftime("%d/%m/%Y")
        prev = row["previsao"]
        inf = row["limite_inferior"]
        sup = row["limite_superior"]
        linhas.append(f"   {data_str}: {prev:>8.0f} unidades [{inf:.0f} - {sup:.0f}]")

    linhas.append("")
    linhas.append("-" * 60)
    linhas.append("PARÂMETROS DO MODELO")
    linhas.append("-" * 60)
    for param, valor in resultado.parametros.items():
        if valor is not None:
            if isinstance(valor, float):
                linhas.append(f"   {param}: {valor:.4f}")
            else:
                linhas.append(f"   {param}: {valor}")

    linhas.append("")
    linhas.append("=" * 60)

    return "\n".join(linhas)


def previsoes_para_dataframe(
    resultados: Dict[str, ResultadoPrevisao]
) -> pd.DataFrame:
    """
    Converte resultados de previsão para DataFrame consolidado.

    Args:
        resultados: Dict com ResultadoPrevisao por produto

    Returns:
        DataFrame com todas as previsões
    """
    registros = []

    for produto_id, resultado in resultados.items():
        for _, row in resultado.previsoes.iterrows():
            registros.append({
                "produto_id": produto_id,
                "loja_id": resultado.loja_id or "TODAS",
                "data": row["data"],
                "previsao": row["previsao"],
                "limite_inferior": row["limite_inferior"],
                "limite_superior": row["limite_superior"],
                "metodo": resultado.metodo,
                "mape": resultado.metricas["mape"]
            })

    return pd.DataFrame(registros)


if __name__ == "__main__":
    # Teste do módulo
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from data_generator import gerar_historico_vendas, PRODUTOS

    print("Gerando dados de teste...")
    historico = gerar_historico_vendas(dias=60)  # Mais dias para previsão

    print("\n" + "=" * 60)
    print("   TESTE DE PREVISÃO DE DEMANDA")
    print("=" * 60)

    previsor = PrevisaoDemanda(historico)

    # Testar para um produto
    produto_teste = "PROD_005"  # Refrigerante (alta sazonalidade)
    nome_produto = PRODUTOS[produto_teste]["nome"]

    print(f"\nProduto: {nome_produto}")

    # Simple Exponential Smoothing
    print("\n--- Simple Exponential Smoothing ---")
    resultado_ses = previsor.simple_exponential_smoothing(produto_teste, horizonte=7)
    print(f"MAPE: {resultado_ses.metricas['mape']:.2f}%")

    # Holt Linear
    print("\n--- Holt Linear ---")
    resultado_holt = previsor.holt_linear(produto_teste, horizonte=7)
    print(f"MAPE: {resultado_holt.metricas['mape']:.2f}%")

    # Holt-Winters
    print("\n--- Holt-Winters ---")
    resultado_hw = previsor.holt_winters(produto_teste, horizonte=7, periodo_sazonal=7)
    print(f"MAPE: {resultado_hw.metricas['mape']:.2f}%")

    # Auto seleção
    print("\n--- Auto Seleção ---")
    resultado_auto = previsor.auto_selecionar_metodo(produto_teste, horizonte=7)
    print(f"Melhor método: {resultado_auto.metodo}")
    print(f"MAPE: {resultado_auto.metricas['mape']:.2f}%")

    # Relatório completo
    print("\n" + gerar_relatorio_previsao(resultado_auto, nome_produto))

    # Previsão para todos os produtos
    print("\n" + "=" * 60)
    print("   PREVISÃO PARA TODOS OS PRODUTOS")
    print("=" * 60)

    resultados_todos = previsor.prever_todos_produtos(horizonte=7, metodo="auto")

    for prod_id, res in resultados_todos.items():
        nome = PRODUTOS[prod_id]["nome"]
        demanda_total = res.previsoes["previsao"].sum()
        print(f"\n{nome}:")
        print(f"   Método: {res.metodo}")
        print(f"   MAPE: {res.metricas['mape']:.2f}%")
        print(f"   Demanda prevista (7 dias): {demanda_total:.0f} unidades")
