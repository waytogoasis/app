// EventLogger.gs
//
// Funcionalidade Principal: Registra eventos específicos do sistema para análise e depuração.
//
// Descrição: Registro de eventos discretos de negócio (início de simulação, conclusão de avaliação,
//            interações). Persistidos na aba `EventLog`.
//
// Integrações:
// - Google Planilha (aba `EventLog`): Armazenamento dos eventos.
// - UserService.gs (wtg* helpers): camada de CRUD reutilizável.
//
// Funções Principais:
// - `logEvent(eventType, eventDetails, userId)`: Registra um evento no log.
// - `getEventsByType(eventType)`: Retorna eventos de um tipo específico.
// - `getEventsByUser(userId)`: Retorna eventos associados a um usuário.

var EVENT_LOG_SHEET = 'EventLog';
var EVENT_LOG_HEADERS = ['ID', 'EventType', 'Details', 'UserID', 'CriadoEm', 'AtualizadoEm'];

function logEvent(eventType, eventDetails, userId) {
  try {
    return wtgCreateRecord_(EVENT_LOG_SHEET, EVENT_LOG_HEADERS, {
      EventType: eventType || 'evento',
      Details: typeof eventDetails === 'object' ? JSON.stringify(eventDetails) : String(eventDetails || ''),
      UserID: userId || ''
    }, { required: [] });
  } catch (error) {
    Logger.log("Erro em logEvent: " + error.message);
    throw error;
  }
}

function getEventsByType(eventType) {
  return wtgReadObjects_(EVENT_LOG_SHEET)
    .filter(function (e) { return String(e.EventType || e.eventtype || '') === String(eventType); });
}

function getEventsByUser(userId) {
  return wtgReadObjects_(EVENT_LOG_SHEET)
    .filter(function (e) { return String(e.UserID || e.userid || '') === String(userId); });
}
