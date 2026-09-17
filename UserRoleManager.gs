// UserRoleManager.gs
//
// Funcionalidade Principal: Gerencia a atribuição e modificação de papéis de usuário.
//
// Descrição: Permite atribuir/remover papéis (aluno, professor, admin, coordenador) aos usuários.
//            Os papéis são armazenados no campo `Role` da aba `Usuarios` (múltiplos, separados por vírgula).
//
// Integrações:
// - Google Planilha (aba `Usuarios`): Armazenamento dos papéis.
// - UserService.gs (wtg* helpers): leitura/atualização dos usuários.
// - PermissionService.gs: Consome os papéis para verificar permissões.
//
// Funções Principais:
// - `assignRoleToUser(userId, role)`: Atribui um papel (sem duplicar).
// - `removeRoleFromUser(userId, role)`: Remove um papel.
// - `getUserRoles(userId)`: Retorna a lista de papéis de um usuário.
// - `listAvailableRoles()`: Lista os papéis disponíveis no sistema.

var USUARIOS_SHEET = 'Usuarios';
var AVAILABLE_ROLES = ['aluno', 'professor', 'coordenador', 'admin'];

function urm_userRaw_(userId) {
  try {
    return wtgReadObjects_(USUARIOS_SHEET)
      .filter(function (u) { return String(u.ID || u.id || '') === String(userId); })[0] || null;
  } catch (error) {
    Logger.log("Erro em urm_userRaw_: " + error.message);
    throw error;
  }
}

function urm_parseRoles_(raw) {
  try {
    return String(raw || '').split(',').map(function (r) { return r.trim(); }).filter(Boolean);
  } catch (error) {
    Logger.log("Erro em urm_parseRoles_: " + error.message);
    throw error;
  }
}

function getUserRoles(userId) {
  var u = urm_userRaw_(userId);
  return u ? urm_parseRoles_(u.Role || u.role) : [];
}

function listAvailableRoles() {
  try {
    return AVAILABLE_ROLES.slice();
  } catch (error) {
    Logger.log("Erro em listAvailableRoles: " + error.message);
    throw error;
  }
}

function assignRoleToUser(userId, role) {
  try {
    role = String(role || '').trim().toLowerCase();
    if (AVAILABLE_ROLES.indexOf(role) === -1) return { success: false, message: 'Papel invalido: ' + role };
    var u = urm_userRaw_(userId);
    if (!u) return { success: false, message: 'Usuario nao encontrado.' };
    var roles = urm_parseRoles_(u.Role || u.role);
    if (roles.indexOf(role) === -1) roles.push(role);
    return wtgUpdateRecordById_(USUARIOS_SHEET, userId, { Role: roles.join(',') });
  } catch (error) {
    Logger.log("Erro em assignRoleToUser: " + error.message);
    throw error;
  }
}

function removeRoleFromUser(userId, role) {
  try {
    role = String(role || '').trim().toLowerCase();
    var u = urm_userRaw_(userId);
    if (!u) return { success: false, message: 'Usuario nao encontrado.' };
    var roles = urm_parseRoles_(u.Role || u.role).filter(function (r) { return r !== role; });
    return wtgUpdateRecordById_(USUARIOS_SHEET, userId, { Role: roles.join(',') });
  } catch (error) {
    Logger.log("Erro em removeRoleFromUser: " + error.message);
    throw error;
  }
}
