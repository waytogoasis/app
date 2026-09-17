// StudentFeedbackManager.gs
//
// Funcionalidade Principal: Gerencia o registro e a recuperação de feedback dos alunos.
//
// Descrição: Este script permite que professores registrem observações e feedback qualitativo
//            sobre o desempenho e comportamento dos alunos durante as simulações.
//
// Integrações:
// - Google Planilha (aba `FeedbackAlunos`): Armazenamento do feedback.
// - UserService.gs (wtg* helpers): camada de CRUD reutilizável.
// - AlunoService.gs: Para associar feedback a alunos específicos.
//
// Funções Principais:
// - `recordFeedback(alunoId, simulationId, feedbackText, teacherId)`: Registra um feedback para um aluno.
// - `getFeedbackByAluno(alunoId)`: Retorna todo o feedback de um aluno.
// - `getFeedbackBySimulation(simulationId)`: Retorna o feedback de uma simulação específica.

var FEEDBACK_ALUNOS_SHEET = 'FeedbackAlunos';
var FEEDBACK_ALUNOS_HEADERS = ['ID', 'AlunoID', 'SimulacaoID', 'Feedback', 'ProfessorID', 'CriadoEm', 'AtualizadoEm'];

function recordFeedback(alunoId, simulationId, feedbackText, teacherId) {
  try {
    if (String(alunoId || '').trim() === '') return { success: false, message: 'alunoId obrigatorio.' };
    if (String(feedbackText || '').trim() === '') return { success: false, message: 'Feedback vazio.' };
    return wtgCreateRecord_(FEEDBACK_ALUNOS_SHEET, FEEDBACK_ALUNOS_HEADERS, {
      AlunoID: alunoId,
      SimulacaoID: simulationId || '',
      Feedback: feedbackText,
      ProfessorID: teacherId || ''
    }, { required: ['AlunoID'] });
  } catch (error) {
    Logger.log("Erro em recordFeedback: " + error.message);
    throw error;
  }
}

function getFeedbackByAluno(alunoId) {
  try {
    return wtgReadObjects_(FEEDBACK_ALUNOS_SHEET)
      .filter(function (r) { return String(r.AlunoID || r.alunoid || '') === String(alunoId); });
  } catch (error) {
    Logger.log("Erro em getFeedbackByAluno: " + error.message);
    throw error;
  }
}

function getFeedbackBySimulation(simulationId) {
  try {
    return wtgReadObjects_(FEEDBACK_ALUNOS_SHEET)
      .filter(function (r) { return String(r.SimulacaoID || r.simulacaoid || '') === String(simulationId); });
  } catch (error) {
    Logger.log("Erro em getFeedbackBySimulation: " + error.message);
    throw error;
  }
}
