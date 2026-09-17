// NotificationService.gs
//
// Funcionalidade Principal: Gerencia o envio de notificações dentro do sistema.
//
// Descrição: Envia notificações in-app (persistidas na aba `Notificacoes`), por e-mail
//            (via EmailService) e registra todas as notificações para rastreabilidade.
//
// Integrações:
// - EmailService.gs: notificações por e-mail.
// - UserService.gs (wtg* helpers): persistência e busca de e-mail do usuário.
//
// Funções Principais:
// - `sendInAppNotification(userId, message, type)`: Cria uma notificação in-app não lida.
// - `sendEmailNotification(userId, subject, message)`: Envia notificação por e-mail ao usuário.
// - `logNotification(userId, message, type)`: Registra a notificação (canal/estado).

var NOTIFICACOES_SHEET = 'Notificacoes';
var NOTIFICACOES_HEADERS = ['ID', 'UserID', 'Mensagem', 'Tipo', 'Canal', 'Status', 'CriadoEm', 'AtualizadoEm'];

function logNotification(userId, message, type, canal, status) {
  return wtgCreateRecord_(NOTIFICACOES_SHEET, NOTIFICACOES_HEADERS, {
    UserID: userId || '', Mensagem: message || '', Tipo: type || 'info',
    Canal: canal || 'in_app', Status: status || 'enviada'
  }, { required: [] });
}

function sendInAppNotification(userId, message, type) {
  try {
    if (String(userId || '').trim() === '') return { success: false, message: 'userId obrigatorio.' };
    var reg = logNotification(userId, message, type || 'info', 'in_app', 'nao_lida');
    return { success: true, data: reg.data };
  } catch (error) {
    Logger.log("Erro em sendInAppNotification: " + error.message);
    throw error;
  }
}

function ns_userEmail_(userId) {
  try {
    var u = wtgReadObjects_('Usuarios').filter(function (x) { return String(x.ID || x.id) === String(userId); })[0];
    return u ? (u.Email || u.email || '') : '';
  } catch (error) {
    Logger.log("Erro em ns_userEmail_: " + error.message);
    throw error;
  }
}

function sendEmailNotification(userId, subject, message) {
  var email = ns_userEmail_(userId);
  var enviado = false;
  if (email && typeof sendEmail === 'function') {
    var res = sendEmail(email, subject, message);
    enviado = !!(res && res.success);
  }
  logNotification(userId, message, 'email', 'email', enviado ? 'enviada' : 'falha');
  return { success: true, data: { destinatario: email, enviado: enviado } };
}
