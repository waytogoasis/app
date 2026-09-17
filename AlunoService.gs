// AlunoService.gs
//
// Funcionalidade Principal: Gerencia as operações CRUD para alunos no sistema.
//
// Descrição: Este script oferece funções para interagir com a aba 'Alunos' da Google Planilha,
//            permitindo o cadastro de novos alunos, a recuperação de informações de alunos
//            existentes, a atualização de seus dados e a remoção de registros.
//            É fundamental para manter o registro dos participantes das simulações.
//
// Integrações:
// - Google Planilha (aba 'Alunos'): Todas as operações de dados são realizadas nesta aba.
// - SpreadsheetUtils.gs: Utiliza funções auxiliares para manipulação da planilha.
//
// Funções Principais:
// - `createAluno(alunoData)`: Adiciona um novo aluno à planilha.
// - `getAlunoById(alunoId)`: Retorna os dados de um aluno específico.
// - `updateAluno(alunoId, newAlunoData)`: Atualiza as informações de um aluno.
// - `deleteAluno(alunoId)`: Remove um aluno da planilha.
// - `getAllAlunos()`: Retorna uma lista de todos os alunos cadastrados.

function createAluno(alunoData) {
  return wtgCreateRecord_('Alunos', ['ID', 'Nome', 'TurmaID', 'Status', 'CriadoEm', 'AtualizadoEm'], alunoData, {
    required: ['Nome'],
    defaults: { Status: 'ativo' }
  });
}

function getAlunoById(alunoId) {
  return wtgFindRecordById_('Alunos', alunoId);
}

function updateAluno(alunoId, newAlunoData) {
  return wtgUpdateRecordById_('Alunos', alunoId, newAlunoData || {});
}

function deleteAluno(alunoId) {
  return wtgUpdateRecordById_('Alunos', alunoId, { Status: 'inativo', Ativo: false });
}

function getAllAlunos() {
  try {
    return wtgReadObjects_('Alunos').filter(function (aluno) {
      return String(aluno.Status || aluno.status || 'ativo').toLowerCase() !== 'inativo';
    });
  } catch (error) {
    Logger.log("Erro em getAllAlunos: " + error.message);
    throw error;
  }
}
