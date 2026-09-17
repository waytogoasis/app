// Logger.gs
//
// Funcionalidade Principal: Fornece um sistema de registro (logging) para o sistema.
//
// Descrição: Este script permite registrar eventos, erros e informações de depuração
//            em um local centralizado, como uma aba específica da Google Planilha
//            ("Logs") ou no Cloud Logging do Google Cloud Platform. É essencial para
//            monitorar o comportamento do sistema e diagnosticar problemas.
//
// Integrações:
// - Google Planilha (aba `Logs`): Armazenamento de logs (best-effort).
// - Todos os Services: Utilizam este serviço para registrar eventos.
//
// Funções Principais:
// - `logInfo(message)`: Registra uma mensagem informativa.
// - `logWarning(message)`: Registra uma mensagem de aviso.
// - `logError(message, errorObject)`: Registra uma mensagem de erro com detalhes.
// - `logDebug(message)`: Registra uma mensagem de depuração.
//
// Observações: A escrita na planilha é best-effort; o console nativo é sempre usado.

var LOGGER_SHEET = 'Logs';
var LOGGER_DEBUG_ENABLED = false;

function logWrite_(level, message, detail) {
  var line = '[' + level + '] ' + message + (detail ? ' :: ' + detail : '');
  try {
    if (level === 'ERROR') console.error(line);
    else if (level === 'WARNING') console.warn(line);
    else console.log(line);
  } catch (e) {}
  // Persistência best-effort na aba Logs (não interrompe o fluxo em caso de falha).
  try {
    if (typeof appendRow === 'function') {
      appendRow(LOGGER_SHEET, [new Date().toISOString(), level, String(message), detail || '']);
    }
  } catch (e2) {}
}

function logInfo(message) { logWrite_('INFO', message); }
function logWarning(message) { logWrite_('WARNING', message); }
function logError(message, errorObject) {
  var detail = errorObject ? (errorObject.stack || errorObject.message || String(errorObject)) : '';
  logWrite_('ERROR', message, detail);
}
function logDebug(message) {
  if (LOGGER_DEBUG_ENABLED) logWrite_('DEBUG', message);
}
