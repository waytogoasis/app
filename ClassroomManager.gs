// ClassroomManager.gs
//
// Funcionalidade Principal: Gerencia a criação, edição e associação de turmas de alunos.
//
// Descrição: Organiza os alunos em turmas, facilitando a gestão pedagógica e a aplicação de
//            simulações e avaliações em grupos. Interage com a aba `Turmas` e com a aba `Alunos`
//            (campo TurmaID) para a composição.
//
// Integrações:
// - Google Planilha (aba `Turmas`): Armazenamento das informações das turmas.
// - AlunoService.gs: associação aluno↔turma via campo TurmaID do aluno.
// - UserService.gs (wtg* helpers): camada de CRUD reutilizável.
//
// Funções Principais:
// - `createClassroom(className, teacherId)`: Cria uma nova turma.
// - `addStudentToClassroom(studentId, classId)`: Associa um aluno a uma turma (atualiza TurmaID do aluno).
// - `getClassroomStudents(classId)`: Retorna a lista de alunos de uma turma.
// - `getAllClassrooms()`: Retorna todas as turmas cadastradas.

var TURMAS_SHEET = 'Turmas';
var TURMAS_HEADERS = ['ID', 'Nome', 'ProfessorID', 'AnoLetivoID', 'Status', 'CriadoEm', 'AtualizadoEm'];

function createClassroom(className, teacherId) {
  try {
    if (String(className || '').trim() === '') return { success: false, message: 'Nome da turma obrigatorio.' };
    return wtgCreateRecord_(TURMAS_SHEET, TURMAS_HEADERS, {
      Nome: className, ProfessorID: teacherId || '', AnoLetivoID: '', Status: 'ativa'
    }, { required: ['Nome'] });
  } catch (error) {
    Logger.log("Erro em createClassroom: " + error.message);
    throw error;
  }
}

function addStudentToClassroom(studentId, classId) {
  if (typeof updateAluno === 'function') {
    var res = updateAluno(studentId, { TurmaID: classId });
    if (res && res.success) return res;
  }
  // Fallback: atualiza diretamente a aba Alunos.
  return wtgUpdateRecordById_('Alunos', studentId, { TurmaID: classId });
}

function getClassroomStudents(classId) {
  try {
    var alunos = (typeof getAllAlunos === 'function') ? getAllAlunos() : wtgReadObjects_('Alunos');
    return alunos.filter(function (a) { return String(a.TurmaID || a.turmaid || '') === String(classId); });
  } catch (error) {
    Logger.log("Erro em getClassroomStudents: " + error.message);
    throw error;
  }
}

function getAllClassrooms() {
  return wtgReadObjects_(TURMAS_SHEET);
}
