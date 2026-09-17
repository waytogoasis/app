// AdvancedReporting.gs
//
// Funcionalidade Principal: Fornece funcionalidades avançadas para a geração de relatórios.
//
// Descrição: Relatórios complexos — comparação entre turmas, longitudinal de aluno por período,
//            e relatórios personalizados por configuração.
//
// Integrações:
// - RelatorioService / PerformanceAnalyzer / ClassroomPerformanceAnalyzer.
//
// Funções Principais:
// - `generateComparativeClassroomReport(classroomId1, classroomId2)`: Compara duas turmas.
// - `generateLongitudinalStudentReport(alunoId, startYear, endYear)`: Relatório longitudinal de um aluno.
// - `generateCustomReport(reportConfig)`: Relatório com base em configuração personalizada.

function generateComparativeClassroomReport(classroomId1, classroomId2) {
  try {
    var comparacao = (typeof compareClassroomPerformance === 'function')
      ? compareClassroomPerformance(classroomId1, classroomId2, 'Total') : null;
    return {
      success: true,
      data: {
        tipo: 'comparativo_turmas',
        comparacao: comparacao,
        tendencia1: (typeof getClassroomProgressTrend === 'function') ? getClassroomProgressTrend(classroomId1, 'Total') : null,
        tendencia2: (typeof getClassroomProgressTrend === 'function') ? getClassroomProgressTrend(classroomId2, 'Total') : null,
        geradoEm: new Date().toISOString()
      }
    };
  } catch (error) {
    Logger.log("Erro em generateComparativeClassroomReport: " + error.message);
    throw error;
  }
}

function generateLongitudinalStudentReport(alunoId, startYear, endYear) {
  try {
    var prog = (typeof getProgressoAluno === 'function') ? getProgressoAluno(alunoId) : { serie: [] };
    var serie = (prog.serie || []).filter(function (ponto) {
      if (!ponto.data) return true;
      var ano = new Date(ponto.data).getFullYear();
      var okIni = startYear ? ano >= Number(startYear) : true;
      var okFim = endYear ? ano <= Number(endYear) : true;
      return okIni && okFim;
    });
    return {
      success: true,
      data: {
        tipo: 'longitudinal_aluno', alunoId: alunoId,
        periodo: { de: startYear || null, ate: endYear || null },
        pontos: serie.length, serie: serie,
        evolucao: serie.length >= 2 ? Math.round((serie[serie.length - 1].total - serie[0].total) * 100) / 100 : 0,
        geradoEm: new Date().toISOString()
      }
    };
  } catch (error) {
    Logger.log("Erro em generateLongitudinalStudentReport: " + error.message);
    throw error;
  }
}

function generateCustomReport(reportConfig) {
  try {
    reportConfig = reportConfig || {};
    var tipo = reportConfig.tipo || 'geral';
    var resultado = {};
    if (tipo === 'turma' && reportConfig.classId) {
      resultado = (typeof getAveragePerformanceByClassroom === 'function') ? getAveragePerformanceByClassroom(reportConfig.classId) : {};
    } else if (tipo === 'aluno' && reportConfig.alunoId) {
      resultado = (typeof getProgressoAluno === 'function') ? getProgressoAluno(reportConfig.alunoId) : {};
    } else {
      resultado = (typeof getEstatisticasGerais === 'function') ? getEstatisticasGerais() : {};
    }
    return { success: true, data: { tipo: tipo, config: reportConfig, resultado: resultado, geradoEm: new Date().toISOString() } };
  } catch (error) {
    Logger.log("Erro em generateCustomReport: " + error.message);
    throw error;
  }
}
