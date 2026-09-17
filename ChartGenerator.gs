// ChartGenerator.gs
//
// Funcionalidade Principal: Prepara dados e gera configurações para gráficos.
//
// Descrição: Coleta e formata dados de pontuações/turmas para consumo por bibliotecas de
//            gráficos (Chart.js) no frontend, via ChartConfigBuilder.
//
// Integrações:
// - PontuacaoService / RelatorioService / ClassroomPerformanceAnalyzer / AvaliacaoService.
// - ChartConfigBuilder.gs: montagem da configuração final.
//
// Funções Principais:
// - `getAlunoProgressChartData(alunoId)`: Série de progresso do aluno (linha).
// - `getOverallPerformanceChartData()`: Desempenho por turma (barras).
// - `getRubricPerformanceChartData(rubricId)`: Desempenho por critério da rubrica (pizza/barras).
// - `generateChartConfig(chartType, data, options)`: Gera a configuração completa do gráfico.

function getAlunoProgressChartData(alunoId) {
  try {
    var prog = (typeof getProgressoAluno === 'function') ? getProgressoAluno(alunoId) : { serie: [] };
    var serie = prog.serie || [];
    return {
      labels: serie.map(function (p) { return String(p.data || '').slice(0, 10); }),
      datasets: [{ label: 'Progresso (média)', data: serie.map(function (p) { return p.total; }) }]
    };
  } catch (error) {
    Logger.log("Erro em getAlunoProgressChartData: " + error.message);
    throw error;
  }
}

function getOverallPerformanceChartData() {
  try {
    var turmas = (typeof getTopPerformingClassrooms === 'function') ? getTopPerformingClassrooms('Total', 10) : [];
    return {
      labels: turmas.map(function (t) { return t.nome || t.classId; }),
      datasets: [{ label: 'Média por turma', data: turmas.map(function (t) { return t.media; }) }]
    };
  } catch (error) {
    Logger.log("Erro em getOverallPerformanceChartData: " + error.message);
    throw error;
  }
}

function getRubricPerformanceChartData(rubricId) {
  try {
    // Distribuição média dos critérios a partir das pontuações registradas.
    var pontuacoes = (typeof wtgReadObjects_ === 'function') ? wtgReadObjects_('Pontuacoes') : [];
    var acc = {};
    pontuacoes.forEach(function (p) {
      var ind; try { ind = JSON.parse(p.Pontuacoes || '{}'); } catch (e) { ind = {}; }
      Object.keys(ind).forEach(function (k) { (acc[k] = acc[k] || []).push(Number(ind[k]) || 0); });
    });
    var labels = Object.keys(acc);
    var data = labels.map(function (k) {
      var arr = acc[k];
      return Math.round(arr.reduce(function (a, b) { return a + b; }, 0) / arr.length * 100) / 100;
    });
    return { rubricId: rubricId || null, labels: labels, datasets: [{ label: 'Média por critério', data: data }] };
  } catch (error) {
    Logger.log("Erro em getRubricPerformanceChartData: " + error.message);
    throw error;
  }
}

function generateChartConfig(chartType, data, options) {
  data = data || { labels: [], datasets: [] };
  switch (chartType) {
    case 'line': return buildLineChartConfig(data.labels, data.datasets, options);
    case 'bar': return buildBarChartConfig(data.labels, data.datasets, options);
    case 'pie':
      var pieData = (data.datasets && data.datasets[0]) ? data.datasets[0].data : (data.data || []);
      return buildPieChartConfig(data.labels, pieData, options);
    default: return buildBarChartConfig(data.labels, data.datasets, options);
  }
}
