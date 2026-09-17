// AuditLogService.gs
//
// Funcionalidade Principal: Mantém um registro de auditoria detalhado de todas as operações críticas do sistema.
//
// Descrição: Este script é responsável por registrar todas as ações que modificam dados sensíveis
//            ou configurações do sistema, incluindo quem realizou a ação, quando e quais foram as mudanças.
//            É essencial para conformidade, segurança e rastreabilidade.
//
// Integrações:
// - Google Planilha (aba `AuditLog`): Armazenamento dos registros de auditoria.
// - SpreadsheetUtils.gs: Para interagir com a planilha de auditoria.
// - SessionManager.gs: Para obter informações do usuário que realizou a ação.
//
// Funções Principais:
// - `logAudit(userId, action, entity, entityId, changes)`: Registra uma entrada no log de auditoria.
// - `getAuditLog(filter)`: Retorna entradas do log de auditoria com base em filtros.
//
// Observações: O log de auditoria é uma ferramenta crítica para a segurança e a governança de dados.

/**
 * Registra uma entrada no log de auditoria.
 * @param {string|number} userId - ID do usuário que realizou a ação
 * @param {string} action - Ação realizada (CREATE, UPDATE, DELETE, LOGIN, LOGOUT, etc.)
 * @param {string} entity - Entidade afetada (Usuarios, Alunos, Simulacoes, etc.)
 * @param {string|number} entityId - ID da entidade afetada
 * @param {Object} changes - Objeto com as mudanças realizadas (antes/depois)
 * @return {Object} Resultado da operação
 */
function logAudit(userId, action, entity, entityId, changes) {
  try {
    var timestamp = new Date();
    var sessionUser = null;
    
    // Tenta obter informações adicionais do usuário se possível
    try {
      if (typeof getCurrentSessionUser !== 'undefined') {
        sessionUser = getCurrentSessionUser();
      } else if (typeof getUserById !== 'undefined' && userId) {
        sessionUser = getUserById(userId);
      }
    } catch (e) {
      Logger.log("Aviso: não foi possível obter dados do usuário para auditoria: " + e.message);
    }

    var auditEntry = {
      Timestamp: timestamp.toISOString(),
      UserID: userId || 'SYSTEM',
      Username: sessionUser ? (sessionUser.Username || sessionUser.username || sessionUser.Nome || 'Unknown') : 'Unknown',
      Action: String(action || '').toUpperCase(),
      Entity: String(entity || ''),
      EntityID: entityId || '',
      Changes: JSON.stringify(changes || {}),
      IPAddress: getClientIpAddress_(),
      UserAgent: getUserAgent_()
    };

    // Tenta registrar na planilha AuditLog
    try {
      if (typeof appendRow === 'function') {
        appendRow('AuditLog', auditEntry);
      } else if (typeof wtgCreateRecord_ === 'function') {
        wtgCreateRecord_('AuditLog', 
          ['ID', 'Timestamp', 'UserID', 'Username', 'Action', 'Entity', 'EntityID', 'Changes', 'IPAddress', 'UserAgent'],
          auditEntry,
          { required: ['Action', 'Entity'] }
        );
      } else {
        throw new Error("Nenhuma função de escrita em planilha disponível");
      }
    } catch (writeError) {
      // Fallback: registra no Logger se não conseguir escrever na planilha
      Logger.log("AUDIT LOG: " + JSON.stringify(auditEntry));
      Logger.log("Erro ao gravar auditoria na planilha: " + writeError.message);
    }

    // Também registra no StructuredLogService se disponível (redundância)
    try {
      if (typeof StructuredLogService !== 'undefined' && StructuredLogService.logEvent) {
        StructuredLogService.logEvent('AUDIT', {
          userId: userId,
          action: action,
          entity: entity,
          entityId: entityId,
          changes: changes
        });
      }
    } catch (e) {
      // Ignora se StructuredLogService não estiver disponível
    }

    return { success: true, timestamp: timestamp.toISOString() };
  } catch (error) {
    Logger.log("Erro em logAudit: " + error.message);
    // Não lança erro para não quebrar operações críticas
    return { success: false, error: error.message };
  }
}

/**
 * Recupera entradas do log de auditoria com base em filtros.
 * @param {Object} filter - Objeto com critérios de filtro (userId, action, entity, entityId, startDate, endDate)
 * @return {Array<Object>} Array de entradas de auditoria que correspondem ao filtro
 */
function getAuditLog(filter) {
  try {
    filter = filter || {};
    var allLogs = [];

    // Tenta ler da planilha AuditLog
    try {
      if (typeof getDataAsObjects === 'function') {
        allLogs = getDataAsObjects('AuditLog');
      } else if (typeof wtgReadObjects_ === 'function') {
        allLogs = wtgReadObjects_('AuditLog');
      } else {
        throw new Error("Nenhuma função de leitura de planilha disponível");
      }
    } catch (readError) {
      Logger.log("Erro ao ler log de auditoria da planilha: " + readError.message);
      return [];
    }

    if (!allLogs || allLogs.length === 0) {
      return [];
    }

    // Aplica filtros
    var filtered = allLogs.filter(function(entry) {
      // Filtro por UserID
      if (filter.userId && String(entry.UserID || entry.userId || '') !== String(filter.userId)) {
        return false;
      }

      // Filtro por Action
      if (filter.action && String(entry.Action || '').toUpperCase() !== String(filter.action).toUpperCase()) {
        return false;
      }

      // Filtro por Entity
      if (filter.entity && String(entry.Entity || '').toLowerCase() !== String(filter.entity).toLowerCase()) {
        return false;
      }

      // Filtro por EntityID
      if (filter.entityId && String(entry.EntityID || entry.entityId || '') !== String(filter.entityId)) {
        return false;
      }

      // Filtro por data de início
      if (filter.startDate) {
        try {
          var entryDate = new Date(entry.Timestamp || entry.timestamp);
          var startDate = new Date(filter.startDate);
          if (entryDate < startDate) {
            return false;
          }
        } catch (e) {
          // Se não conseguir parsear data, mantém o registro
        }
      }

      // Filtro por data final
      if (filter.endDate) {
        try {
          var entryDate = new Date(entry.Timestamp || entry.timestamp);
          var endDate = new Date(filter.endDate);
          if (entryDate > endDate) {
            return false;
          }
        } catch (e) {
          // Se não conseguir parsear data, mantém o registro
        }
      }

      return true;
    });

    // Ordena por timestamp decrescente (mais recentes primeiro)
    filtered.sort(function(a, b) {
      var dateA = new Date(a.Timestamp || a.timestamp || 0);
      var dateB = new Date(b.Timestamp || b.timestamp || 0);
      return dateB - dateA;
    });

    // Limita resultados se especificado
    if (filter.limit && filter.limit > 0) {
      filtered = filtered.slice(0, filter.limit);
    }

    return filtered;
  } catch (error) {
    Logger.log("Erro em getAuditLog: " + error.message);
    return [];
  }
}

/**
 * Funções auxiliares privadas para coleta de metadados
 */
function getClientIpAddress_() {
  try {
    // Em contexto web app, tenta obter IP do request
    if (typeof Session !== 'undefined' && Session.getTemporaryActiveUserKey) {
      return Session.getTemporaryActiveUserKey();
    }
    return 'N/A';
  } catch (e) {
    return 'N/A';
  }
}

function getUserAgent_() {
  try {
    // Em contexto web, poderia ser capturado do frontend
    return 'Apps Script';
  } catch (e) {
    return 'N/A';
  }
}
