// StudentGroupManager.gs
//
// Funcionalidade Principal: Gerencia a formação e a composição de grupos de alunos para as simulações.
//
// Descrição: Este script permite criar, modificar e gerenciar grupos de alunos, o que é essencial
//            para as atividades de simulação onde os alunos atuam em conjunto (ex: "tremzinho").
//
// Integrações:
// - Google Planilha (aba `Grupos`): Armazenamento das composições dos grupos.
// - UserService.gs (wtg* helpers): camada de CRUD reutilizável.
// - AlunoService.gs: Para obter a lista de alunos disponíveis.
//
// Funções Principais:
// - `createGroup(groupName, studentIds)`: Cria um novo grupo com alunos específicos.
// - `assignStudentsToGroup(studentIds, groupId)`: Adiciona alunos (sem duplicar) a um grupo existente.
// - `getGroupMembers(groupId)`: Retorna os IDs de alunos que fazem parte de um grupo.
// - `getAllGroups()`: Retorna todos os grupos formados (com membros parseados).

var GRUPOS_SHEET = 'Grupos';
var GRUPOS_HEADERS = ['ID', 'Nome', 'Membros', 'CriadoEm', 'AtualizadoEm'];

function sgm_parseMembers_(raw) {
  try {
    try { var m = JSON.parse(raw || '[]'); return Array.isArray(m) ? m : []; } catch (e) { return []; }
  } catch (error) {
    Logger.log("Erro em sgm_parseMembers_: " + error.message);
    throw error;
  }
}

function createGroup(groupName, studentIds) {
  if (String(groupName || '').trim() === '') return { success: false, message: 'Nome do grupo obrigatorio.' };
  var membros = (studentIds || []).map(String);
  return wtgCreateRecord_(GRUPOS_SHEET, GRUPOS_HEADERS, {
    Nome: groupName, Membros: JSON.stringify(membros)
  }, { required: ['Nome'] });
}

function assignStudentsToGroup(studentIds, groupId) {
  var found = wtgFindRecordById_(GRUPOS_SHEET, groupId);
  if (!found.success) return { success: false, message: 'Grupo nao encontrado.' };
  var membros = sgm_parseMembers_(found.data.Membros);
  (studentIds || []).forEach(function (id) { if (membros.indexOf(String(id)) === -1) membros.push(String(id)); });
  return wtgUpdateRecordById_(GRUPOS_SHEET, groupId, { Membros: JSON.stringify(membros) });
}

function getGroupMembers(groupId) {
  var found = wtgFindRecordById_(GRUPOS_SHEET, groupId);
  return found.success ? sgm_parseMembers_(found.data.Membros) : [];
}

function getAllGroups() {
  try {
    return wtgReadObjects_(GRUPOS_SHEET).map(function (g) {
      g.MembrosParsed = sgm_parseMembers_(g.Membros);
      return g;
    });
  } catch (error) {
    Logger.log("Erro em getAllGroups: " + error.message);
    throw error;
  }
}
