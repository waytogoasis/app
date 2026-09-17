// SimulationMetrics.gs
//
// Funcionalidade Principal: Calcula e agrega métricas de desempenho das simulações.
//
// Descrição: Processa os dados de simulações e pontuações para extrair métricas significativas:
//            tempo médio de reação, número de infrações e desempenho geral.
//
// Integrações:
// - PontuacaoService.gs: pontuações detalhadas (indicadores em JSON).
// - SimulacaoService.gs: informações sobre as simulações.
//
// Funções Principais:
// - `getAverageReactionTime(simulationId)`: Tempo médio de reação (indicador `tempo_reacao`) na simulação.
// - `getInfractionCount(simulationId)`: Total de infrações (indicador `infracoes`) na simulação.
// - `getOverallSimulationPerformance()`: Métricas agregadas de todas as simulações.
// - `getSimulationMetricsByAluno(alunoId)`: Métricas agregadas de um aluno.

function sm_indicator_(pontuacoes, key) {
  try {
    var vals = (pontuacoes || []).map(function (p) {
      var ind = p.IndicadoresParsed || p.PontuacoesParsed || {};
      return Number(ind[key]);
    }).filter(function (n) { return !isNaN(n); });
    return vals;
  } catch (error) {
    Logger.log("Erro em sm_indicator_: " + error.message);
    throw error;
  }
}

function getAverageReactionTime(simulationId) {
  try {
    var pts = (typeof getPontuacoesBySimulacao === 'function') ? getPontuacoesBySimulacao(simulationId) : [];
    var vals = sm_indicator_(pts, 'tempo_reacao');
    if (!vals.length) return 0;
    return Math.round(vals.reduce(function (a, b) { return a + b; }, 0) / vals.length * 100) / 100;
  } catch (error) {
    Logger.log("Erro em getAverageReactionTime: " + error.message);
    throw error;
  }
}

function getInfractionCount(simulationId) {
  try {
    var pts = (typeof getPontuacoesBySimulacao === 'function') ? getPontuacoesBySimulacao(simulationId) : [];
    return sm_indicator_(pts, 'infracoes').reduce(function (a, b) { return a + b; }, 0);
  } catch (error) {
    Logger.log("Erro em getInfractionCount: " + error.message);
    throw error;
  }
}

function getOverallSimulationPerformance() {
  try {
    var simulacoes = (typeof getAllSimulations === 'function') ? getAllSimulations() : [];
    var pontuacoes = (typeof wtgReadObjects_ === 'function') ? wtgReadObjects_('Pontuacoes') : [];
    var totais = pontuacoes.map(function (p) { return Number(p.Total) || 0; });
    var media = totais.length ? Math.round(totais.reduce(function (a, b) { return a + b; }, 0) / totais.length * 100) / 100 : 0;
    return {
      totalSimulacoes: simulacoes.length,
      finalizadas: simulacoes.filter(function (s) { return String(s.Status) === 'finalizada'; }).length,
      totalAvaliacoes: pontuacoes.length,
      mediaGeral: media
    };
  } catch (error) {
    Logger.log("Erro em getOverallSimulationPerformance: " + error.message);
    throw error;
  }
}

function getSimulationMetricsByAluno(alunoId) {
  try {
    var sims = (typeof getSimulationsByAluno === 'function') ? getSimulationsByAluno(alunoId) : [];
    var pts = (typeof getPontuacoesByAluno === 'function') ? getPontuacoesByAluno(alunoId) : [];
    var totais = pts.map(function (p) { return Number(p.Total) || 0; });
    var media = totais.length ? Math.round(totais.reduce(function (a, b) { return a + b; }, 0) / totais.length * 100) / 100 : 0;
    return {
      alunoId: alunoId,
      simulacoes: sims.length,
      avaliacoes: pts.length,
      mediaGeral: media,
      melhor: totais.length ? Math.max.apply(null, totais) : 0,
      pior: totais.length ? Math.min.apply(null, totais) : 0
    };
  } catch (error) {
    Logger.log("Erro em getSimulationMetricsByAluno: " + error.message);
    throw error;
  }
}
