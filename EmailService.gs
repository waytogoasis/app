// EmailService.gs
//
// Funcionalidade Principal: Envia e-mails para usuários do sistema.
//
// Descrição: Fornece funções para enviar notificações e informações via MailApp, com tratamento
//            de erros e suporte a templates (via NotificationTemplates).
//
// Integrações:
// - MailApp (Apps Script): Serviço nativo para envio de e-mails.
// - NotificationTemplates.gs: para e-mails baseados em template.
// - Logger.gs: registro de falhas de envio.
//
// Funções Principais:
// - `sendEmail(recipient, subject, body)`: Envia um e-mail simples (retorna envelope de status).
// - `sendTemplatedEmail(recipient, templateName, data)`: Envia um e-mail renderizado de um template.

function sendEmail(recipient, subject, body) {
  try {
    if (!recipient || String(recipient).indexOf('@') === -1) {
      return { success: false, message: 'Destinatário inválido.' };
    }
    try {
      MailApp.sendEmail(recipient, subject || '(sem assunto)', String(body || ''));
      return { success: true, data: { recipient: recipient, subject: subject } };
    } catch (error) {
      if (typeof logError === 'function') logError('Falha ao enviar e-mail para ' + recipient, error);
      return { success: false, message: (error && error.message) ? error.message : String(error) };
    }
  } catch (error) {
    Logger.log("Erro em sendEmail: " + error.message);
    throw error;
  }
}

function sendTemplatedEmail(recipient, templateName, data) {
  var subject = (data && data.subject) ? data.subject : templateName;
  var body;
  if (typeof getTemplate === 'function' && typeof renderTemplate === 'function') {
    var tpl = getTemplate(templateName, (data && data.language) || 'pt-BR');
    body = tpl ? renderTemplate(tpl.content || tpl.Content || tpl, data || {}) : '';
  } else {
    body = (data && data.body) || '';
  }
  if (!body) return { success: false, message: 'Template não encontrado: ' + templateName };
  return sendEmail(recipient, subject, body);
}
