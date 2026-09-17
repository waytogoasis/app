// ClassroomPerformanceAnalyzer.gs
//
// Funcionalidade Principal: Analisa o desempenho agregado de turmas.
//
// Descrição: Compara o desempenho médio de turmas, identifica turmas com melhor/pior desempenho
//            e analisa tendências de progresso em nível de turma.
//
// Integrações:
// - PontuacaoService.gs / AlunoService.gs / ClassroomManager.gs: fontes de dados.
// - PerformanceAnalyzer.gs: helpers estatísticos `wtgMean_`.
//
// Funções Principais:
// - `getAverageScoreByClassroom(classId, metric)`: Pontuação média da turma (Total ou indicador).
// - `compareClassroomPerformance(classId1, classId2, metric)`: Compara duas turmas.
// - `getClassroomProgressTrend(classId, metric)`: Tendência de progresso da turma ao longo do tempo.
// - `getTopPerformingClassrooms(metric, limit)`: Ranqueia turmas por desempenho.

function cpa_pontuacoesDaTurma_(classId) {
  try {
    var alunos = (typeof getClassroomStudents === 'function') ? getClassroomStudents(classId) : [];
    var ids = {};
    alunos.forEach(function (a) { ids[String(a.ID || a.id)] = true; });
    var pts = (typeof wtgReadObjects_ === 'function') ? wtgReadObjects_('Pontuacoes') : [];
    return pts.filter(function (p) { return ids[String(p.AlunoID || p.alunoid)]; }).map(function (p) {
      try { p.IndicadoresParsed = JSON.parse(p.Pontuacoes || '{}'); } catch (e) { p.IndicadoresParsed = {}; }
      return p;
    });
  } catch (error) {
    Logger.log("Erro em cpa_pontuacoesDaTurma_: " + error.message);
    throw error;
  }
}

function cpa_valor_(p, metric) {
  if (!metric || metric === 'Total') return Number(p.Total) || 0;
  var ind = p.IndicadoresParsed || {};
  return Number(ind[metric]) || 0;
}

function getAverageScoreByClassroom(classId, metric) {
  var pts = cpa_pontuacoesDaTurma_(classId);
  var vals = pts.map(function (p) { return cpa_valor_(p, metric); });
  return {
    classId: classId, metric: metric || 'Total',
    amostras: vals.length,
    media: Math.round(wtgMean_(vals) * 100) / 100
  };
}

function compareClassroomPerformance(classId1, classId2, metric) {
  var a = getAverageScoreByClassroom(classId1, metric);
  var b = getAverageScoreByClassroom(classId2, metric);
  return {
    metric: metric || 'Total',
    turma1: a, turma2: b,
    diferenca: Math.round((a.media - b.media) * 100) / 100,
    melhor: a.media === b.media ? 'empate' : (a.media > b.media ? classId1 : classId2)
  };
}

function getClassroomProgressTrend(classId, metric) {
  try {
    var pts = cpa_pontuacoesDaTurma_(classId)
      .sort(function (x, y) { return new Date(x.CriadoEm) - new Date(y.CriadoEm); });
    // Agrupa por data (yyyy-MM-dd) e calcula a média da turma em cada dia.
    var porDia = {};
    pts.forEach(function (p) {
      var dia = String(p.CriadoEm || '').slice(0, 10) || 'sem-data';
      (porDia[dia] = porDia[dia] || []).push(cpa_valor_(p, metric));
    });
    var serie = Object.keys(porDia).sort().map(function (dia) {
      return { data: dia, media: Math.round(wtgMean_(porDia[dia]) * 100) / 100 };
    });
    var delta = serie.length >= 2 ? Math.round((serie[serie.length - 1].media - serie[0].media) * 100) / 100 : 0;
    return {
      classId: classId, metric: metric || 'Total', serie: serie, delta: delta,
      tendencia: serie.length < 2 ? 'sem_dados' : (delta > 2 ? 'melhorando' : (delta < -2 ? 'declinando' : 'estavel'))
    };
  } catch (error) {
    Logger.log("Erro em getClassroomProgressTrend: " + error.message);
    throw error;
  }
}

function getTopPerformingClassrooms(metric, limit) {
  try {
    var turmas = (typeof getAllClassrooms === 'function') ? getAllClassrooms() : [];
    return turmas
      .map(function (t) {
        var avg = getAverageScoreByClassroom(t.ID || t.id, metric);
        return { classId: t.ID || t.id, nome: t.Nome || t.nome, media: avg.media, amostras: avg.amostras };
      })
      .filter(function (t) { return t.amostras > 0; })
      .sort(function (a, b) { return b.media - a.media; })
      .slice(0, limit || 5);
  } catch (error) {
    Logger.log("Erro em getTopPerformingClassrooms: " + error.message);
    throw error;
  }
}
