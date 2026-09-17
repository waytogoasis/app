// PermissionService.gs
//
// Funcionalidade Principal: Implementa o controle de acesso baseado em papel (Role-Based Access Control - RBAC).
//
// Descrição: Este script define e verifica as permissões dos usuários com base em seus papéis
//            (ex: Administrador, Professor, Aluno). Ele garante que apenas usuários autorizados
//            possam executar certas ações ou acessar determinadas partes da aplicação.
//
// Integrações:
// - Google Planilha (aba `Usuarios`): Onde os papéis dos usuários são definidos.
// - AuthService.gs: Utiliza informações do usuário logado para verificar permissões.
// - Todos os Services: Devem chamar este serviço para verificar permissões antes de executar ações sensíveis.
//
// Funções Principais:
// - `hasRole(user, role)`: Verifica se um usuário possui um papel específico.
// - `can(user, action)`: Verifica se um usuário tem permissão para realizar uma ação.
// - `getRoles(user)`: Retorna os papéis de um usuário.
//
// Observações: A implementação de RBAC é crucial para a segurança e integridade do sistema,
//              especialmente em um ambiente multiusuário.

/**
 * Definição de papéis e suas permissões
 * Sistema RBAC (Role-Based Access Control)
 */
var ROLES = {
  ADMIN: 'admin',
  ADMINISTRATOR: 'administrator',
  PROFESSOR: 'professor',
  TEACHER: 'teacher',
  ALUNO: 'aluno',
  STUDENT: 'student',
  USER: 'user'
};

/**
 * Mapa de permissões por papel
 * Cada ação é mapeada para os papéis que podem executá-la
 */
var PERMISSIONS = {
  // Administração de sistema
  'system.configure': ['admin', 'administrator'],
  'system.backup': ['admin', 'administrator'],
  'system.reset': ['admin', 'administrator'],
  'system.viewLogs': ['admin', 'administrator'],
  
  // Gestão de usuários
  'users.create': ['admin', 'administrator'],
  'users.edit': ['admin', 'administrator'],
  'users.delete': ['admin', 'administrator'],
  'users.view': ['admin', 'administrator', 'professor', 'teacher'],
  'users.forceLogout': ['admin', 'administrator'],
  
  // Gestão de alunos
  'alunos.create': ['admin', 'administrator', 'professor', 'teacher'],
  'alunos.edit': ['admin', 'administrator', 'professor', 'teacher'],
  'alunos.delete': ['admin', 'administrator', 'professor', 'teacher'],
  'alunos.view': ['admin', 'administrator', 'professor', 'teacher', 'aluno', 'student'],
  
  // Gestão de turmas
  'turmas.create': ['admin', 'administrator', 'professor', 'teacher'],
  'turmas.edit': ['admin', 'administrator', 'professor', 'teacher'],
  'turmas.delete': ['admin', 'administrator', 'professor', 'teacher'],
  'turmas.view': ['admin', 'administrator', 'professor', 'teacher'],
  
  // Simulações
  'simulacoes.create': ['admin', 'administrator', 'professor', 'teacher'],
  'simulacoes.edit': ['admin', 'administrator', 'professor', 'teacher'],
  'simulacoes.delete': ['admin', 'administrator', 'professor', 'teacher'],
  'simulacoes.view': ['admin', 'administrator', 'professor', 'teacher', 'aluno', 'student'],
  'simulacoes.participate': ['aluno', 'student'],
  
  // Pontuações
  'pontuacoes.create': ['admin', 'administrator', 'professor', 'teacher'],
  'pontuacoes.edit': ['admin', 'administrator', 'professor', 'teacher'],
  'pontuacoes.delete': ['admin', 'administrator', 'professor', 'teacher'],
  'pontuacoes.view': ['admin', 'administrator', 'professor', 'teacher', 'aluno', 'student'],
  
  // Relatórios
  'relatorios.generate': ['admin', 'administrator', 'professor', 'teacher'],
  'relatorios.view': ['admin', 'administrator', 'professor', 'teacher'],
  'relatorios.export': ['admin', 'administrator', 'professor', 'teacher'],
  
  // API
  'api.read': ['admin', 'administrator', 'professor', 'teacher'],
  'api.write': ['admin', 'administrator', 'professor', 'teacher'],
  
  // Auditoria
  'audit.view': ['admin', 'administrator']
};

/**
 * Normaliza o papel do usuário para formato consistente
 * @param {string} role - Papel do usuário
 * @return {string} Papel normalizado
 */
function normalizeRole_(role) {
  if (!role) return '';
  var normalized = String(role).toLowerCase().trim();
  
  // Mapeia variações para papéis canônicos
  if (normalized === 'administrator' || normalized === 'administrador') return 'admin';
  if (normalized === 'teacher' || normalized === 'professor') return 'professor';
  if (normalized === 'student' || normalized === 'estudante') return 'aluno';
  
  return normalized;
}

/**
 * Verifica se um usuário possui um papel específico.
 * @param {Object|string|number} user - Objeto usuário, ID de usuário ou papel direto
 * @param {string} role - Papel a verificar (admin, professor, aluno, etc.)
 * @return {boolean} true se o usuário possui o papel, false caso contrário
 */
function hasRole(user, role) {
  try {
    if (!user || !role) return false;
    
    var userRole = '';
    
    // Se user é um objeto
    if (typeof user === 'object') {
      userRole = user.role || user.Role || user.papel || user.Papel || '';
      
      // Se não tem papel no objeto, tenta buscar do banco
      if (!userRole && (user.id || user.ID || user.userId)) {
        try {
          var userId = user.id || user.ID || user.userId;
          if (typeof getUserById === 'function') {
            var fullUser = getUserById(userId);
            if (fullUser) {
              userRole = fullUser.Role || fullUser.role || '';
            }
          }
        } catch (e) {
          Logger.log("Erro ao buscar papel do usuário: " + e.message);
        }
      }
    } 
    // Se user é um ID numérico, busca o usuário
    else if (typeof user === 'number' || !isNaN(Number(user))) {
      try {
        if (typeof getUserById === 'function') {
          var fullUser = getUserById(user);
          if (fullUser) {
            userRole = fullUser.Role || fullUser.role || '';
          }
        }
      } catch (e) {
        Logger.log("Erro ao buscar usuário por ID: " + e.message);
      }
    }
    // Se user é uma string, assume que é o papel diretamente
    else if (typeof user === 'string') {
      userRole = user;
    }
    
    // Normaliza ambos os papéis para comparação
    var normalizedUserRole = normalizeRole_(userRole);
    var normalizedRequestedRole = normalizeRole_(role);
    
    return normalizedUserRole === normalizedRequestedRole;
  } catch (error) {
    Logger.log("Erro em hasRole: " + error.message);
    return false;
  }
}

/**
 * Verifica se um usuário tem permissão para realizar uma ação.
 * @param {Object|string|number} user - Objeto usuário, ID ou papel
 * @param {string} action - Ação a verificar (ex: 'users.create', 'alunos.edit')
 * @return {boolean} true se o usuário tem permissão, false caso contrário
 */
function can(user, action) {
  try {
    if (!user || !action) return false;
    
    // Obtém os papéis do usuário
    var userRoles = getRoles(user);
    if (!userRoles || userRoles.length === 0) return false;
    
    // Verifica se a ação existe nas permissões
    var allowedRoles = PERMISSIONS[action];
    if (!allowedRoles || allowedRoles.length === 0) {
      Logger.log("Ação não mapeada no sistema de permissões: " + action);
      return false;
    }
    
    // Verifica se algum papel do usuário está autorizado para a ação
    for (var i = 0; i < userRoles.length; i++) {
      var normalizedUserRole = normalizeRole_(userRoles[i]);
      for (var j = 0; j < allowedRoles.length; j++) {
        if (normalizedUserRole === normalizeRole_(allowedRoles[j])) {
          return true;
        }
      }
    }
    
    return false;
  } catch (error) {
    Logger.log("Erro em can: " + error.message);
    return false;
  }
}

/**
 * Retorna os papéis de um usuário (array para suportar múltiplos papéis no futuro).
 * @param {Object|string|number} user - Objeto usuário, ID ou papel
 * @return {Array<string>} Array com os papéis do usuário
 */
function getRoles(user) {
  try {
    if (!user) return [];
    
    var userRole = '';
    
    // Se user é um objeto
    if (typeof user === 'object') {
      userRole = user.role || user.Role || user.papel || user.Papel || '';
      
      // Se não tem papel no objeto, tenta buscar do banco
      if (!userRole && (user.id || user.ID || user.userId)) {
        try {
          var userId = user.id || user.ID || user.userId;
          if (typeof getUserById === 'function') {
            var fullUser = getUserById(userId);
            if (fullUser) {
              userRole = fullUser.Role || fullUser.role || '';
            }
          }
        } catch (e) {
          Logger.log("Erro ao buscar papel do usuário: " + e.message);
        }
      }
    }
    // Se user é um ID numérico, busca o usuário
    else if (typeof user === 'number' || !isNaN(Number(user))) {
      try {
        if (typeof getUserById === 'function') {
          var fullUser = getUserById(user);
          if (fullUser) {
            userRole = fullUser.Role || fullUser.role || '';
          }
        }
      } catch (e) {
        Logger.log("Erro ao buscar usuário por ID: " + e.message);
      }
    }
    // Se user é uma string, assume que é o papel diretamente
    else if (typeof user === 'string') {
      userRole = user;
    }
    
    if (!userRole) return [];
    
    // Retorna array (preparado para suporte a múltiplos papéis no futuro)
    return [normalizeRole_(userRole)];
  } catch (error) {
    Logger.log("Erro em getRoles: " + error.message);
    return [];
  }
}

/**
 * Verifica se o usuário é administrador.
 * @param {Object|string|number} user - Objeto usuário, ID ou papel
 * @return {boolean} true se é administrador
 */
function isAdmin(user) {
  return hasRole(user, 'admin') || hasRole(user, 'administrator');
}

/**
 * Verifica se o usuário é professor.
 * @param {Object|string|number} user - Objeto usuário, ID ou papel
 * @return {boolean} true se é professor
 */
function isProfessor(user) {
  return hasRole(user, 'professor') || hasRole(user, 'teacher');
}

/**
 * Verifica se o usuário é aluno.
 * @param {Object|string|number} user - Objeto usuário, ID ou papel
 * @return {boolean} true se é aluno
 */
function isAluno(user) {
  return hasRole(user, 'aluno') || hasRole(user, 'student');
}

/**
 * Valida permissão ou lança erro (útil para guards de entrada de função).
 * @param {Object|string|number} user - Objeto usuário, ID ou papel
 * @param {string} action - Ação a verificar
 * @throws {Error} Se o usuário não tiver permissão
 */
function requirePermission(user, action) {
  if (!can(user, action)) {
    var userRoles = getRoles(user).join(', ') || 'unknown';
    throw new Error('Permissão negada: usuário com papel(is) [' + userRoles + '] não pode executar ação "' + action + '"');
  }
}
