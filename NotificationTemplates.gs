// NotificationTemplates.gs
//
// Funcionalidade Principal: Gerencia templates para diferentes tipos de notificações.
//
// Descrição: Armazena e recupera templates de texto para notificações, com substituição de
//            variáveis no formato {{chave}}. Templates embutidos por padrão, sobreponíveis por planilha.
//
// Integrações:
// - Google Planilha (aba `NotificationTemplates`): Armazenamento dos templates.
// - NotificationService.gs / EmailService.gs: consomem os templates.
//
// Funções Principais:
// - `getTemplate(templateId, language)`: Retorna o template (planilha → embutido).
// - `renderTemplate(templateContent, data)`: Preenche `{{chave}}` com os dados fornecidos.
// - `createTemplate(templateId, content)`: Cria ou atualiza um template.

var NOTIFICATION_TEMPLATES_SHEET = 'NotificationTemplates';
var NOTIFICATION_TEMPLATES_HEADERS = ['ID', 'Language', 'Content', 'CriadoEm', 'AtualizadoEm'];
var NOTIFICATION_TEMPLATES_DEFAULT = {
  'boas_vindas:pt-BR': 'Olá {{nome}}, bem-vindo ao Way To Go!',
  'relatorio_pronto:pt-BR': 'Olá {{nome}}, seu relatório de {{periodo}} está pronto.',
  'lembrete_simulacao:pt-BR': 'Olá {{nome}}, há uma simulação agendada para {{data}}.'
};

function nt_key_(templateId, language) { return templateId + ':' + (language || 'pt-BR'); }

function getTemplate(templateId, language) {
  try {
    language = language || 'pt-BR';
    var raw = wtgReadObjects_(NOTIFICATION_TEMPLATES_SHEET).filter(function (t) {
      return String(t.ID) === String(templateId) && String(t.Language || 'pt-BR') === language;
    })[0];
    if (raw) return { id: templateId, language: language, content: raw.Content, fonte: 'planilha' };
    var def = NOTIFICATION_TEMPLATES_DEFAULT[nt_key_(templateId, language)];
    if (def) return { id: templateId, language: language, content: def, fonte: 'default' };
    return null;
  } catch (error) {
    Logger.log("Erro em getTemplate: " + error.message);
    throw error;
  }
}

function renderTemplate(templateContent, data) {
  try {
    var content = (templateContent && templateContent.content) ? templateContent.content : String(templateContent || '');
    data = data || {};
    return content.replace(/\{\{\s*(\w+)\s*\}\}/g, function (m, key) {
      return data[key] !== undefined ? String(data[key]) : '';
    });
  } catch (error) {
    Logger.log("Erro em renderTemplate: " + error.message);
    throw error;
  }
}

function createTemplate(templateId, content, language) {
  try {
    if (String(templateId || '').trim() === '') return { success: false, message: 'templateId obrigatorio.' };
    language = language || 'pt-BR';
    var existing = wtgReadObjects_(NOTIFICATION_TEMPLATES_SHEET).filter(function (t) {
      return String(t.ID) === String(templateId) && String(t.Language || 'pt-BR') === language;
    })[0];
    if (existing) return wtgUpdateRecordById_(NOTIFICATION_TEMPLATES_SHEET, existing.ID, { Content: content || '' });
    return wtgCreateRecord_(NOTIFICATION_TEMPLATES_SHEET, NOTIFICATION_TEMPLATES_HEADERS, {
      ID: templateId, Language: language, Content: content || ''
    }, { required: ['ID'] });
  } catch (error) {
    Logger.log("Erro em createTemplate: " + error.message);
    throw error;
  }
}
