// AdminFunctions.gs
//
// Funcionalidade Principal: Contém funções administrativas para gerenciamento do sistema.
//
// Descrição: Este script agrupa funções que são tipicamente acessíveis apenas por usuários
//            com perfil de administrador. Inclui tarefas como resetar dados, gerenciar
//            configurações sensíveis, e realizar manutenção no sistema.
//
// Integrações:
// - ConfigService.gs: Para gerenciar configurações do sistema.
// - UserService.gs, AlunoService.gs, SimulacaoService.gs, PontuacaoService.gs: Para operações
//   de CRUD em massa ou sensíveis.
// - PermissionService.gs: Para garantir que apenas administradores possam executar estas funções.
// - Logger.gs: Para registrar todas as ações administrativas.
//
// Funções Principais:
// - `resetAllData()`: Apaga todos os dados de todas as abas da planilha (com confirmação).
// - `backupData()`: Cria um backup da planilha principal.
// - `changeSystemSetting(key, value)`: Altera uma configuração crítica do sistema.
// - `forceLogoutUser(userId)`: Força o logout de um usuário específico.
//
// Observações: Funções neste script devem ser protegidas por rigorosas verificações de permissão.

/**
 * Apaga todos os dados de todas as abas da planilha (OPERAÇÃO DESTRUTIVA).
 * Requer confirmação explícita e permissões de administrador.
 * @param {Object} currentUser - Usuário executando a operação (deve ser admin)
 * @param {string} confirmationToken - Token de confirmação (deve ser "CONFIRM_RESET_ALL_DATA")
 * @return {Object} Resultado da operação
 */
function resetAllData_(currentUser, confirmationToken) {
  try {
    // Verificação de permissão
    if (typeof requirePermission === 'function') {
      requirePermission(currentUser, 'system.reset');
    } else if (typeof isAdmin === 'function' && !isAdmin(currentUser)) {
      throw new Error("Permissão negada: apenas administradores podem resetar dados");
    }

    // Verificação de confirmação (proteção contra execução acidental)
    if (confirmationToken !== 'CONFIRM_RESET_ALL_DATA') {
      throw new Error("Token de confirmação inválido. Esta é uma operação destrutiva que requer confirmação explícita.");
    }

    // Registra auditoria ANTES da operação
    try {
      if (typeof logAudit === 'function') {
        logAudit(
          currentUser.id || currentUser.ID || currentUser.userId || 'admin',
          'RESET_ALL_DATA',
          'System',
          'ALL',
          { confirmation: confirmationToken, timestamp: new Date().toISOString() }
        );
      }
    } catch (auditError) {
      Logger.log("Aviso: não foi possível registrar auditoria: " + auditError.message);
    }

    var ss = getSpreadsheet_();
    var sheets = ss.getSheets();
    var resetCount = 0;
    var errors = [];

    // Lista de abas críticas que NÃO devem ser resetadas (mantém estrutura)
    var protectedSheets = ['Settings', 'Config', 'AuditLog'];

    for (var i = 0; i < sheets.length; i++) {
      var sheet = sheets[i];
      var sheetName = sheet.getName();

      // Pula abas protegidas
      if (protectedSheets.indexOf(sheetName) !== -1) {
        Logger.log("Aba protegida não resetada: " + sheetName);
        continue;
      }

      try {
        var lastRow = sheet.getLastRow();
        if (lastRow > 1) { // Mantém cabeçalho (linha 1)
          sheet.deleteRows(2, lastRow - 1);
          resetCount++;
          Logger.log("Dados resetados na aba: " + sheetName);
        }
      } catch (sheetError) {
        errors.push({ sheet: sheetName, error: sheetError.message });
        Logger.log("Erro ao resetar aba " + sheetName + ": " + sheetError.message);
      }
    }

    return {
      success: errors.length === 0,
      sheetsReset: resetCount,
      errors: errors,
      timestamp: new Date().toISOString(),
      performedBy: currentUser.username || currentUser.Username || 'admin'
    };
  } catch (error) {
    Logger.log("Erro em resetAllData: " + error.message);
    throw error;
  }
}

/**
 * Cria um backup completo da planilha principal.
 * @param {Object} currentUser - Usuário executando a operação (deve ser admin)
 * @param {string} [backupName] - Nome opcional para o backup (padrão: "Backup_YYYY-MM-DD_HH-MM")
 * @return {Object} Resultado da operação com ID e URL do backup
 */
function backupData_(currentUser, backupName) {
  try {
    // Verificação de permissão
    if (typeof requirePermission === 'function') {
      requirePermission(currentUser, 'system.backup');
    } else if (typeof isAdmin === 'function' && !isAdmin(currentUser)) {
      throw new Error("Permissão negada: apenas administradores podem criar backups");
    }

    var ss = getSpreadsheet_();
    var now = new Date();
    
    // Gera nome do backup se não fornecido
    if (!backupName) {
      var timestamp = Utilities.formatDate(now, Session.getScriptTimeZone(), "yyyy-MM-dd_HH-mm");
      backupName = "Backup_WayToGo_" + timestamp;
    }

    // Cria cópia da planilha
    var backupFile = DriveApp.getFileById(ss.getId()).makeCopy(backupName);
    var backupId = backupFile.getId();
    var backupUrl = backupFile.getUrl();

    // Tenta mover para pasta de backups (se configurada)
    try {
      var backupFolderId = getSetting('BACKUP_FOLDER_ID');
      if (backupFolderId) {
        var backupFolder = DriveApp.getFolderById(backupFolderId);
        backupFile.moveTo(backupFolder);
      }
    } catch (folderError) {
      Logger.log("Aviso: não foi possível mover backup para pasta específica: " + folderError.message);
    }

    // Registra auditoria
    try {
      if (typeof logAudit === 'function') {
        logAudit(
          currentUser.id || currentUser.ID || currentUser.userId || 'admin',
          'BACKUP_CREATED',
          'System',
          backupId,
          { 
            backupName: backupName, 
            backupUrl: backupUrl,
            timestamp: now.toISOString() 
          }
        );
      }
    } catch (auditError) {
      Logger.log("Aviso: não foi possível registrar auditoria: " + auditError.message);
    }

    return {
      success: true,
      backupId: backupId,
      backupUrl: backupUrl,
      backupName: backupName,
      timestamp: now.toISOString(),
      performedBy: currentUser.username || currentUser.Username || 'admin'
    };
  } catch (error) {
    Logger.log("Erro em backupData: " + error.message);
    throw error;
  }
}

/**
 * Altera uma configuração crítica do sistema.
 * @param {Object} currentUser - Usuário executando a operação (deve ser admin)
 * @param {string} key - Chave da configuração
 * @param {*} value - Novo valor da configuração
 * @param {string} [description] - Descrição da configuração (opcional)
 * @return {Object} Resultado da operação
 */
function changeSystemSetting_(currentUser, key, value, description) {
  try {
    // Verificação de permissão
    if (typeof requirePermission === 'function') {
      requirePermission(currentUser, 'system.configure');
    } else if (typeof isAdmin === 'function' && !isAdmin(currentUser)) {
      throw new Error("Permissão negada: apenas administradores podem alterar configurações do sistema");
    }

    if (!key) {
      throw new Error("Chave da configuração não pode ser vazia");
    }

    // Obtém valor anterior para auditoria
    var oldValue = null;
    try {
      if (typeof getSetting === 'function') {
        oldValue = getSetting(key);
      }
    } catch (e) {
      Logger.log("Aviso: não foi possível obter valor anterior da configuração");
    }

    // Atualiza configuração
    var result;
    if (typeof setSetting === 'function') {
      result = setSetting(key, value, description);
    } else {
      // Fallback: grava em Script Properties
      PropertiesService.getScriptProperties().setProperty(key, String(value));
      result = { success: true, method: 'ScriptProperties' };
    }

    // Registra auditoria
    try {
      if (typeof logAudit === 'function') {
        logAudit(
          currentUser.id || currentUser.ID || currentUser.userId || 'admin',
          'SETTING_CHANGED',
          'Settings',
          key,
          { 
            key: key,
            oldValue: oldValue,
            newValue: value,
            description: description,
            timestamp: new Date().toISOString()
          }
        );
      }
    } catch (auditError) {
      Logger.log("Aviso: não foi possível registrar auditoria: " + auditError.message);
    }

    return {
      success: true,
      key: key,
      oldValue: oldValue,
      newValue: value,
      timestamp: new Date().toISOString(),
      performedBy: currentUser.username || currentUser.Username || 'admin'
    };
  } catch (error) {
    Logger.log("Erro em changeSystemSetting: " + error.message);
    throw error;
  }
}

/**
 * Força o logout de um usuário específico (invalida suas sessões).
 * @param {Object} currentUser - Usuário executando a operação (deve ser admin)
 * @param {string|number} targetUserId - ID do usuário a ter logout forçado
 * @param {string} [reason] - Motivo do logout forçado (para auditoria)
 * @return {Object} Resultado da operação
 */
function forceLogoutUser_(currentUser, targetUserId, reason) {
  try {
    // Verificação de permissão
    if (typeof requirePermission === 'function') {
      requirePermission(currentUser, 'users.forceLogout');
    } else if (typeof isAdmin === 'function' && !isAdmin(currentUser)) {
      throw new Error("Permissão negada: apenas administradores podem forçar logout de usuários");
    }

    if (!targetUserId) {
      throw new Error("ID do usuário não pode ser vazio");
    }

    // Verifica se o usuário existe
    var targetUser = null;
    try {
      if (typeof getUserById === 'function') {
        targetUser = getUserById(targetUserId);
        if (!targetUser) {
          throw new Error("Usuário não encontrado: " + targetUserId);
        }
      }
    } catch (e) {
      Logger.log("Aviso: não foi possível verificar existência do usuário: " + e.message);
    }

    var sessionsCleared = 0;

    // Método 1: Limpa sessão via SessionManager
    try {
      if (typeof clearSession === 'function') {
        clearSession();
        sessionsCleared++;
      }
    } catch (e) {
      Logger.log("Aviso: clearSession não disponível ou falhou: " + e.message);
    }

    // Método 2: Invalida tokens de sessão em cache
    try {
      if (typeof CacheService !== 'undefined') {
        var cache = CacheService.getUserCache();
        // Tenta remover chaves comuns de sessão
        var sessionKeys = [
          'WAYTOGO_SESSION',
          'SGTE_CURRENT_SESSION',
          'SESSION_' + targetUserId,
          'USER_SESSION_' + targetUserId
        ];
        sessionKeys.forEach(function(key) {
          try {
            cache.remove(key);
          } catch (e) {}
        });
        sessionsCleared++;
      }
    } catch (e) {
      Logger.log("Aviso: limpeza de cache falhou: " + e.message);
    }

    // Método 3: Marca flag de logout forçado no registro do usuário
    try {
      if (typeof updateUser === 'function') {
        updateUser(targetUserId, {
          ForcedLogout: true,
          ForcedLogoutAt: new Date().toISOString(),
          ForcedLogoutBy: currentUser.id || currentUser.ID || currentUser.userId
        });
      }
    } catch (e) {
      Logger.log("Aviso: não foi possível marcar logout forçado no registro: " + e.message);
    }

    // Registra auditoria
    try {
      if (typeof logAudit === 'function') {
        logAudit(
          currentUser.id || currentUser.ID || currentUser.userId || 'admin',
          'FORCE_LOGOUT',
          'Usuarios',
          targetUserId,
          { 
            targetUserId: targetUserId,
            targetUsername: targetUser ? (targetUser.Username || targetUser.username) : 'unknown',
            reason: reason || 'Não especificado',
            sessionsCleared: sessionsCleared,
            timestamp: new Date().toISOString()
          }
        );
      }
    } catch (auditError) {
      Logger.log("Aviso: não foi possível registrar auditoria: " + auditError.message);
    }

    return {
      success: true,
      targetUserId: targetUserId,
      targetUsername: targetUser ? (targetUser.Username || targetUser.username) : 'unknown',
      sessionsCleared: sessionsCleared,
      reason: reason,
      timestamp: new Date().toISOString(),
      performedBy: currentUser.username || currentUser.Username || 'admin'
    };
  } catch (error) {
    Logger.log("Erro em forceLogoutUser: " + error.message);
    throw error;
  }
}
