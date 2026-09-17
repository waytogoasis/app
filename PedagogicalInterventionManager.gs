// PedagogicalInterventionManager.gs
//
// Funcionalidade Principal: Gerencia o registro e o acompanhamento de intervenções pedagógicas.
//
// Descrição: Permite que professores registrem as intervenções pedagógicas aplicadas a alunos
//            com dificuldades identificadas, monitorando a eficácia e ajustando estratégias.
//
// Integrações:
// - Google Planilha (aba `Intervencoes`): Armazenamento das intervenções.
// - UserService.gs (wtg* helpers): camada de CRUD reutilizável.
// - AlunoService.gs: Para associar intervenções a alunos específicos.
//
// Funções Principais:
// - `recordIntervention(alunoId, interventionType, description, startDate, endDate)`: Registra uma intervenção.
// - `getInterventionsByAluno(alunoId)`: Retorna as intervenções aplicadas a um aluno.
// - `updateInterventionStatus(interventionId, newStatus)`: Atualiza o status de uma intervenção.

var INTERVENCOES_SHEET = 'Intervencoes';
var INTERVENCOES_HEADERS = ['ID', 'AlunoID', 'Tipo', 'Descricao', 'Inicio', 'Fim', 'Status', 'CriadoEm', 'AtualizadoEm'];
var INTERVENCOES_STATUS = ['planejada', 'em_andamento', 'concluida', 'cancelada'];

function recordIntervention(alunoId, interventionType, description, startDate, endDate) {
  try {
    if (String(alunoId || '').trim() === '') return { success: false, message: 'alunoId obrigatorio.' };
    return wtgCreateRecord_(INTERVENCOES_SHEET, INTERVENCOES_HEADERS, {
      AlunoID: alunoId,
      Tipo: interventionType || 'geral',
      Descricao: description || '',
      Inicio: startDate || new Date().toISOString().slice(0, 10),
      Fim: endDate || '',
      Status: 'planejada'
    }, { required: ['AlunoID'] });
  } catch (error) {
    Logger.log("Erro em recordIntervention: " + error.message);
    throw error;
  }
}

function getInterventionsByAluno(alunoId) {
  try {
    return wtgReadObjects_(INTERVENCOES_SHEET)
      .filter(function (r) { return String(r.AlunoID || r.alunoid || '') === String(alunoId); });
  } catch (error) {
    Logger.log("Erro em getInterventionsByAluno: " + error.message);
    throw error;
  }
}

function updateInterventionStatus(interventionId, newStatus) {
  try {
    var status = String(newStatus || '').toLowerCase();
    if (INTERVENCOES_STATUS.indexOf(status) === -1) {
      return { success: false, message: 'Status invalido. Use: ' + INTERVENCOES_STATUS.join(', ') };
    }
    return wtgUpdateRecordById_(INTERVENCOES_SHEET, interventionId, { Status: status });
  } catch (error) {
    Logger.log("Erro em updateInterventionStatus: " + error.message);
    throw error;
  }
}
