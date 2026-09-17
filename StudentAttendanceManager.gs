// StudentAttendanceManager.gs
//
// Funcionalidade Principal: Gerencia o registro e o acompanhamento da frequência dos alunos.
//
// Descrição: Este script permite registrar a presença ou ausência dos alunos nas atividades
//            e simulações. É importante para o controle pedagógico e para a avaliação
//            do engajamento dos alunos no projeto.
//
// Integrações:
// - Google Planilha (aba `Frequencia`): Armazenamento dos registros de frequência.
// - UserService.gs (wtg* helpers): camada de CRUD reutilizável.
// - ClassroomManager.gs: Para gerenciar a frequência por turma.
//
// Funções Principais:
// - `recordAttendance(alunoId, date, status)`: Registra a frequência de um aluno em uma data.
// - `getAttendanceByAluno(alunoId)`: Retorna o histórico de frequência de um aluno.
// - `getAttendanceByClassroom(classId, date)`: Retorna a frequência de uma turma em uma data.

var FREQUENCIA_SHEET = 'Frequencia';
var FREQUENCIA_HEADERS = ['ID', 'AlunoID', 'Data', 'Status', 'TurmaID', 'CriadoEm', 'AtualizadoEm'];

function recordAttendance(alunoId, date, status, turmaId) {
  try {
    if (String(alunoId || '').trim() === '') return { success: false, message: 'alunoId obrigatorio.' };
    var statusNorm = String(status || 'presente').toLowerCase();
    if (['presente', 'ausente', 'justificado'].indexOf(statusNorm) === -1) statusNorm = 'presente';
    return wtgCreateRecord_(FREQUENCIA_SHEET, FREQUENCIA_HEADERS, {
      AlunoID: alunoId,
      Data: date || new Date().toISOString().slice(0, 10),
      Status: statusNorm,
      TurmaID: turmaId || ''
    }, { required: ['AlunoID'] });
  } catch (error) {
    Logger.log("Erro em recordAttendance: " + error.message);
    throw error;
  }
}

function getAttendanceByAluno(alunoId) {
  try {
    return wtgReadObjects_(FREQUENCIA_SHEET)
      .filter(function (r) { return String(r.AlunoID || r.alunoid || '') === String(alunoId); });
  } catch (error) {
    Logger.log("Erro em getAttendanceByAluno: " + error.message);
    throw error;
  }
}

function getAttendanceByClassroom(classId, date) {
  try {
    return wtgReadObjects_(FREQUENCIA_SHEET).filter(function (r) {
      var matchTurma = String(r.TurmaID || r.turmaid || '') === String(classId);
      var matchData = date ? String(r.Data || r.data || '') === String(date) : true;
      return matchTurma && matchData;
    });
  } catch (error) {
    Logger.log("Erro em getAttendanceByClassroom: " + error.message);
    throw error;
  }
}
