// ActivityLogManager.gs
//
// Funcionalidade Principal: Gerencia um log detalhado de todas as atividades e eventos do sistema.
//
// Descrição: Este script é responsável por registrar de forma granular todas as operações
//            significativas que ocorrem no sistema, incluindo interações do usuário, execuções
//            de funções, erros e mudanças de estado. É uma ferramenta essencial para depuração,
//            auditoria e análise forense.
//
// Integrações:
// - Google Planilha (aba `ActivityLog`): Armazenamento de todos os registros de atividade.
// - SpreadsheetUtils.gs: Para interagir com a planilha de log.
// - Logger.gs: Pode ser usado em conjunto para diferentes níveis de detalhe de log.
//
// Funções Principais:
// - `recordSystemActivity(activityType, description, details)`: Registra uma atividade geral do sistema.
// - `getSystemActivities(filter)`: Retorna atividades do sistema com base em filtros.
// - `clearActivityLog(olderThanDate)`: Limpa registros de atividade mais antigos que uma data.
//
// Observações: Um log de atividades bem mantido é inestimável para a manutenção e segurança do sistema.

function recordSystemActivity(activityType, description, details) {
  try {
    var sheet = wtgEnsureSheet_('ActivityLog', ['ID', 'DataHora', 'Tipo', 'Descricao', 'Detalhes']);
    var id = Utilities.getUuid();
    wtgWithWriteLock_('recordSystemActivity', function () {
      sheet.appendRow([
        id,
        new Date().toISOString(),
        String(activityType || 'system'),
        String(description || ''),
        JSON.stringify(details || {})
      ]);
    });
    return { success: true, id: id };
  } catch (error) {
    Logger.log("Erro em recordSystemActivity: " + error.message);
    throw error; // Re-lança para tratamento superior
  }
}

function getSystemActivities(filter) {
  var limit = Math.max(1, Math.min(Number(filter && filter.limit) || 100, 500));
  var items = wtgReadObjects_('ActivityLog');
  if (filter && filter.activityType) {
    items = items.filter(function (item) { return String(item.Tipo || item.tipo) === String(filter.activityType); });
  }
  return items.reverse().slice(0, limit);
}

function clearActivityLog(olderThanDate) {
  try {
    try {
      try {
        var cutoff = olderThanDate ? new Date(olderThanDate) : null;
        if (!cutoff || isNaN(cutoff.getTime())) return { success: false, message: 'Informe uma data valida.' };
        var sheet = wtgGetSpreadsheet_().getSheetByName('ActivityLog');
        if (!sheet || sheet.getLastRow() < 2) return { success: true, removed: 0 };
        var values = sheet.getRange(2, 1, sheet.getLastRow() - 1, sheet.getLastColumn()).getValues();
        var removed = 0;
        wtgWithWriteLock_('clearActivityLog', function () {
          for (var i = values.length - 1; i >= 0; i--) {
            var createdAt = new Date(values[i][1]);
            if (!isNaN(createdAt.getTime()) && createdAt < cutoff) {
              sheet.deleteRow(i + 2);
              removed++;
            }
          }
        });
        return { success: true, removed: removed };
      } catch (error) {
        Logger.log("Erro em clearActivityLog: " + error.message);
        throw error; // Re-lança para tratamento superior
      }
    } catch (error) {
      Logger.log("Erro em clearActivityLog: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em clearActivityLog: " + error.message);
    throw error;
  }
}
