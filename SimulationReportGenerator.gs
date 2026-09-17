// SimulationReportGenerator.gs
//
// Funcionalidade Principal: Gera relatórios detalhados de simulações individuais ou em grupo.
//
// Descrição: Compila dados de simulações (pontuações, feedback, métricas) em relatórios estruturados.
//
// Integrações:
// - SimulacaoService / PontuacaoService / StudentFeedbackManager / SimulationMetrics / StudentGroupManager.
//
// Funções Principais:
// - `generateSingleSimulationReport(simulationId)`: Relatório de uma simulação específica.
// - `generateGroupSimulationReport(groupId)`: Relatório consolidado de um grupo de alunos.
// - `formatSimulationReportData(rawData)`: Estrutura os dados brutos do relatório.

function formatSimulationReportData(rawData) {
  try {
    rawData = rawData || {};
    var pontuacoes = rawData.pontuacoes || [];
    var totais = pontuacoes.map(function (p) { return Number(p.Total) || 0; });
    var media = totais.length ? Math.round(totais.reduce(function (a, b) { return a + b; }, 0) / totais.length * 100) / 100 : 0;
    return {
      simulacao: rawData.simulacao || null,
      totalAvaliacoes: pontuacoes.length,
      mediaPontuacao: media,
      infracoes: rawData.infracoes !== undefined ? rawData.infracoes : null,
      feedback: rawData.feedback || [],
      assessmentInterpretation: {
        assessmentType: 'formative_contextualized',
        rubricRole: 'organizes_human_judgment',
        humanJudgmentRequired: true,
        claims: {
          exclusiveAssessment: false,
          stressReductionDemonstrated: false
        },
        note: 'A rubrica organiza o julgamento docente; este relatório não substitui a revisão humana.'
      },
      geradoEm: new Date().toISOString()
    };
  } catch (error) {
    Logger.log("Erro em formatSimulationReportData: " + error.message);
    throw error;
  }
}

function generateSingleSimulationReport(simulationId) {
  var sim = (typeof getSimulationById === 'function') ? getSimulationById(simulationId) : null;
  if (sim && sim.success === false) return { success: false, message: 'Simulação não encontrada.' };
  var data = formatSimulationReportData({
    simulacao: sim && sim.data ? sim.data : null,
    pontuacoes: (typeof getPontuacoesBySimulacao === 'function') ? getPontuacoesBySimulacao(simulationId) : [],
    infracoes: (typeof getInfractionCount === 'function') ? getInfractionCount(simulationId) : null,
    feedback: (typeof getFeedbackBySimulation === 'function') ? getFeedbackBySimulation(simulationId) : []
  });
  return { success: true, data: data };
}

function generateGroupSimulationReport(groupId) {
  try {
    var membros = (typeof getGroupMembers === 'function') ? getGroupMembers(groupId) : [];
    var porAluno = membros.map(function (alunoId) {
      return {
        alunoId: alunoId,
        metricas: (typeof getSimulationMetricsByAluno === 'function') ? getSimulationMetricsByAluno(alunoId) : null
      };
    });
    var medias = porAluno.map(function (a) { return a.metricas ? a.metricas.mediaGeral : 0; }).filter(function (m) { return m > 0; });
    var mediaGrupo = medias.length ? Math.round(medias.reduce(function (a, b) { return a + b; }, 0) / medias.length * 100) / 100 : 0;
    return {
      success: true,
      data: {
        groupId: groupId,
        membros: porAluno.length,
        mediaGrupo: mediaGrupo,
        detalhes: porAluno,
        assessmentInterpretation: {
          assessmentType: 'formative_contextualized',
          rubricRole: 'organizes_human_judgment',
          humanJudgmentRequired: true,
          claims: {
            exclusiveAssessment: false,
            stressReductionDemonstrated: false
          },
          note: 'A síntese organiza evidências para o julgamento docente; não é uma conclusão automática.'
        },
        geradoEm: new Date().toISOString()
      }
    };
  } catch (error) {
    Logger.log("Erro em generateGroupSimulationReport: " + error.message);
    throw error;
  }
}
