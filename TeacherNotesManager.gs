// TeacherNotesManager.gs
//
// Funcionalidade Principal: Gerencia notas e observações dos professores sobre as aulas e alunos.
//
// Descrição: Permite que professores registrem notas e observações sobre o desempenho dos alunos,
//            incidentes durante as simulações, ou insights pedagógicos.
//
// Integrações:
// - Google Planilha (aba `NotasProfessores`): Armazenamento das notas.
// - UserService.gs (wtg* helpers): camada de CRUD reutilizável.
//
// Funções Principais:
// - `addTeacherNote(teacherId, alunoId, noteText, date)`: Adiciona uma nova nota de professor.
// - `getTeacherNotesByAluno(alunoId)`: Retorna todas as notas associadas a um aluno.
// - `getTeacherNotesByTeacher(teacherId)`: Retorna todas as notas de um professor.

var NOTAS_PROF_SHEET = 'NotasProfessores';
var NOTAS_PROF_HEADERS = ['ID', 'ProfessorID', 'AlunoID', 'Nota', 'Data', 'CriadoEm', 'AtualizadoEm'];

function addTeacherNote(teacherId, alunoId, noteText, date) {
  try {
    if (String(teacherId || '').trim() === '') return { success: false, message: 'teacherId obrigatorio.' };
    if (String(noteText || '').trim() === '') return { success: false, message: 'Nota vazia.' };
    return wtgCreateRecord_(NOTAS_PROF_SHEET, NOTAS_PROF_HEADERS, {
      ProfessorID: teacherId,
      AlunoID: alunoId || '',
      Nota: noteText,
      Data: date || new Date().toISOString().slice(0, 10)
    }, { required: ['ProfessorID'] });
  } catch (error) {
    Logger.log("Erro em addTeacherNote: " + error.message);
    throw error;
  }
}

function getTeacherNotesByAluno(alunoId) {
  try {
    return wtgReadObjects_(NOTAS_PROF_SHEET)
      .filter(function (r) { return String(r.AlunoID || r.alunoid || '') === String(alunoId); });
  } catch (error) {
    Logger.log("Erro em getTeacherNotesByAluno: " + error.message);
    throw error;
  }
}

function getTeacherNotesByTeacher(teacherId) {
  try {
    return wtgReadObjects_(NOTAS_PROF_SHEET)
      .filter(function (r) { return String(r.ProfessorID || r.professorid || '') === String(teacherId); });
  } catch (error) {
    Logger.log("Erro em getTeacherNotesByTeacher: " + error.message);
    throw error;
  }
}
