// DashboardData.gs
//
// Funcionalidade Principal: Prepara e agrega dados para exibição nos dashboards da aplicação.
//
// Descrição: Coleta e agrega informações (alunos, simulações, pontuações, turmas) para os
//            dashboards de administrador, professor e aluno.
//
// Integrações:
// - AlunoService / SimulacaoService / PontuacaoService / RelatorioService / SimulationMetrics.
//
// Funções Principais:
// - `getAdminDashboardData()`: Visão geral + turmas de melhor desempenho.
// - `getProfessorDashboardData(professorId)`: Turmas do professor e seus alunos.
// - `getAlunoDashboardData(alunoId)`: Métricas e progresso do aluno.
// - `getOverallStats()`: Estatísticas gerais do sistema.

function getOverallStats() {
  if (typeof getEstatisticasGerais === 'function') return getEstatisticasGerais();
  var alunos = (typeof getAllAlunos === 'function') ? getAllAlunos() : [];
  return { totalAlunos: alunos.length, totalSimulacoes: 0, totalPontuacoes: 0, mediaGeral: 0 };
}

function getAdminDashboardData() {
  try {
    return {
      stats: getOverallStats(),
      topTurmas: (typeof getTopPerformingClassrooms === 'function') ? getTopPerformingClassrooms('Total', 5) : [],
      totalTurmas: (typeof getAllClassrooms === 'function') ? getAllClassrooms().length : 0,
      geradoEm: new Date().toISOString()
    };
  } catch (error) {
    Logger.log("Erro em getAdminDashboardData: " + error.message);
    throw error;
  }
}

function getProfessorDashboardData(professorId) {
  try {
    var turmas = ((typeof getAllClassrooms === 'function') ? getAllClassrooms() : [])
      .filter(function (t) { return String(t.ProfessorID || t.professorid || '') === String(professorId); });
    var turmasInfo = turmas.map(function (t) {
      var id = t.ID || t.id;
      var alunos = (typeof getClassroomStudents === 'function') ? getClassroomStudents(id) : [];
      var media = (typeof getAveragePerformanceByClassroom === 'function') ? getAveragePerformanceByClassroom(id).mediaTurma : 0;
      return { classId: id, nome: t.Nome || t.nome, alunos: alunos.length, mediaTurma: media };
    });
    return { professorId: professorId, turmas: turmasInfo, totalTurmas: turmasInfo.length, geradoEm: new Date().toISOString() };
  } catch (error) {
    Logger.log("Erro em getProfessorDashboardData: " + error.message);
    throw error;
  }
}

function getAlunoDashboardData(alunoId) {
  try {
    return {
      alunoId: alunoId,
      metricas: (typeof getSimulationMetricsByAluno === 'function') ? getSimulationMetricsByAluno(alunoId) : null,
      progresso: (typeof getProgressoAluno === 'function') ? getProgressoAluno(alunoId) : null,
      conquistas: (typeof getAchievementsByAluno === 'function') ? getAchievementsByAluno(alunoId).length : 0,
      geradoEm: new Date().toISOString()
    };
  } catch (error) {
    Logger.log("Erro em getAlunoDashboardData: " + error.message);
    throw error;
  }
}
