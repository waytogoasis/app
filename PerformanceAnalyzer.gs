// PerformanceAnalyzer.gs
//
// Funcionalidade Principal: Realiza análises aprofundadas sobre o desempenho dos alunos.
//
// Descrição: Usa métodos estatísticos para identificar padrões, pontos fortes e fracos no
//            desempenho dos alunos, comparando resultados entre alunos e turmas.
//
// Integrações:
// - PontuacaoService.gs: Fonte dos dados de pontuação (indicadores em JSON).
// - ClassroomManager.gs: composição das turmas.
//
// Funções Principais:
// - `compareStudentPerformance(alunoId1, alunoId2)`: Compara a média de dois alunos.
// - `identifyWeaknesses(alunoId)`: Identifica os indicadores mais fracos de um aluno.
// - `getAveragePerformanceByClassroom(classId)`: Desempenho médio de uma turma.
// - `analyzeCorrelation(metric1, metric2)`: Correlação de Pearson entre dois indicadores.
//
// Observações: Hospeda os helpers estatísticos compartilhados `wtgStat*_`.

// ===== Helpers estatísticos compartilhados =====
function wtgMean_(values) {
  try {
    var v = (values || []).map(Number).filter(function (n) { return !isNaN(n); });
    if (!v.length) return 0;
    return v.reduce(function (a, b) { return a + b; }, 0) / v.length;
  } catch (error) {
    Logger.log("Erro em wtgMean_: " + error.message);
    throw error;
  }
}

function wtgPearson_(pairs) {
  try {
    // pairs: array de [x, y]
    var xs = pairs.map(function (p) { return Number(p[0]); });
    var ys = pairs.map(function (p) { return Number(p[1]); });
    var n = pairs.length;
    if (n < 2) return 0;
    var mx = wtgMean_(xs), my = wtgMean_(ys);
    var num = 0, dx = 0, dy = 0;
    for (var i = 0; i < n; i++) {
      var a = xs[i] - mx, b = ys[i] - my;
      num += a * b; dx += a * a; dy += b * b;
    }
    if (dx === 0 || dy === 0) return 0;
    return Math.round((num / Math.sqrt(dx * dy)) * 1000) / 1000;
  } catch (error) {
    Logger.log("Erro em wtgPearson_: " + error.message);
    throw error;
  }
}

function pa_alunoMedia_(alunoId) {
  try {
    var pts = (typeof getPontuacoesByAluno === 'function') ? getPontuacoesByAluno(alunoId) : [];
    return Math.round(wtgMean_(pts.map(function (p) { return Number(p.Total) || 0; })) * 100) / 100;
  } catch (error) {
    Logger.log("Erro em pa_alunoMedia_: " + error.message);
    throw error;
  }
}

function pa_indicadoresMedios_(alunoId) {
  try {
    var pts = (typeof getPontuacoesByAluno === 'function') ? getPontuacoesByAluno(alunoId) : [];
    var acc = {};
    pts.forEach(function (p) {
      var ind = p.IndicadoresParsed || p.PontuacoesParsed || {};
      Object.keys(ind).forEach(function (k) { (acc[k] = acc[k] || []).push(Number(ind[k])); });
    });
    var medias = {};
    Object.keys(acc).forEach(function (k) { medias[k] = Math.round(wtgMean_(acc[k]) * 100) / 100; });
    return medias;
  } catch (error) {
    Logger.log("Erro em pa_indicadoresMedios_: " + error.message);
    throw error;
  }
}

function compareStudentPerformance(alunoId1, alunoId2) {
  var m1 = pa_alunoMedia_(alunoId1), m2 = pa_alunoMedia_(alunoId2);
  return {
    aluno1: { alunoId: alunoId1, media: m1 },
    aluno2: { alunoId: alunoId2, media: m2 },
    diferenca: Math.round((m1 - m2) * 100) / 100,
    melhor: m1 === m2 ? 'empate' : (m1 > m2 ? alunoId1 : alunoId2)
  };
}

function identifyWeaknesses(alunoId, limit) {
  try {
    var medias = pa_indicadoresMedios_(alunoId);
    return Object.keys(medias)
      .map(function (k) { return { indicador: k, media: medias[k] }; })
      .sort(function (a, b) { return a.media - b.media; })
      .slice(0, limit || 3);
  } catch (error) {
    Logger.log("Erro em identifyWeaknesses: " + error.message);
    throw error;
  }
}

function getAveragePerformanceByClassroom(classId) {
  try {
    var alunos = (typeof getClassroomStudents === 'function') ? getClassroomStudents(classId) : [];
    var medias = alunos.map(function (a) { return pa_alunoMedia_(a.ID || a.id); }).filter(function (m) { return m > 0; });
    return {
      classId: classId,
      alunos: alunos.length,
      avaliados: medias.length,
      mediaTurma: Math.round(wtgMean_(medias) * 100) / 100
    };
  } catch (error) {
    Logger.log("Erro em getAveragePerformanceByClassroom: " + error.message);
    throw error;
  }
}

function analyzeCorrelation(metric1, metric2) {
  try {
    var pontuacoes = (typeof wtgReadObjects_ === 'function') ? wtgReadObjects_('Pontuacoes') : [];
    var pairs = [];
    pontuacoes.forEach(function (p) {
      var ind; try { ind = JSON.parse(p.Pontuacoes || '{}'); } catch (e) { ind = {}; }
      if (ind[metric1] !== undefined && ind[metric2] !== undefined) pairs.push([ind[metric1], ind[metric2]]);
    });
    return { metric1: metric1, metric2: metric2, amostras: pairs.length, correlacao: wtgPearson_(pairs) };
  } catch (error) {
    Logger.log("Erro em analyzeCorrelation: " + error.message);
    throw error;
  }
}
