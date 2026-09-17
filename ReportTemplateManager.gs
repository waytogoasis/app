// ReportTemplateManager.gs
//
// Funcionalidade Principal: Gerencia templates para a geração de relatórios.
//
// Descrição: Armazena e recupera templates de relatórios (HTML, Markdown ou texto),
//            permitindo relatórios padronizados e personalizáveis.
//
// Integrações:
// - Google Planilha (aba `ReportTemplates`): Armazenamento dos templates.
// - UserService.gs (wtg* helpers): camada de CRUD reutilizável.
// - RelatorioService.gs: Utiliza os templates para formatar os relatórios.
//
// Funções Principais:
// - `getReportTemplate(templateId)`: Retorna o conteúdo de um template (ou default embutido).
// - `createReportTemplate(templateId, templateContent)`: Cria ou atualiza um template.
// - `listAvailableTemplates()`: Lista todos os templates disponíveis (embutidos + persistidos).

var REPORT_TEMPLATES_SHEET = 'ReportTemplates';
var REPORT_TEMPLATES_HEADERS = ['ID', 'Conteudo', 'Formato', 'CriadoEm', 'AtualizadoEm'];
var REPORT_TEMPLATES_DEFAULT = {
  aluno: { formato: 'markdown', conteudo: '# Relatório do Aluno {{nome}}\n\nMédia: {{media}}\nSimulações: {{simulacoes}}' },
  turma: { formato: 'markdown', conteudo: '# Relatório da Turma {{turma}}\n\nMédia da turma: {{mediaTurma}}' }
};

function rtm_findRaw_(templateId) {
  try {
    return wtgReadObjects_(REPORT_TEMPLATES_SHEET)
      .filter(function (t) { return String(t.ID || t.id || '') === String(templateId); })[0] || null;
  } catch (error) {
    Logger.log("Erro em rtm_findRaw_: " + error.message);
    throw error;
  }
}

function getReportTemplate(templateId) {
  var raw = rtm_findRaw_(templateId);
  if (raw) return { id: templateId, conteudo: raw.Conteudo, formato: raw.Formato || 'texto', fonte: 'planilha' };
  if (REPORT_TEMPLATES_DEFAULT[templateId]) {
    return { id: templateId, conteudo: REPORT_TEMPLATES_DEFAULT[templateId].conteudo, formato: REPORT_TEMPLATES_DEFAULT[templateId].formato, fonte: 'default' };
  }
  return null;
}

function createReportTemplate(templateId, templateContent, formato) {
  try {
    if (String(templateId || '').trim() === '') return { success: false, message: 'templateId obrigatorio.' };
    var raw = rtm_findRaw_(templateId);
    if (raw) return wtgUpdateRecordById_(REPORT_TEMPLATES_SHEET, raw.ID, { Conteudo: templateContent || '', Formato: formato || raw.Formato || 'texto' });
    return wtgCreateRecord_(REPORT_TEMPLATES_SHEET, REPORT_TEMPLATES_HEADERS, {
      ID: templateId, Conteudo: templateContent || '', Formato: formato || 'texto'
    }, { required: ['ID'] });
  } catch (error) {
    Logger.log("Erro em createReportTemplate: " + error.message);
    throw error;
  }
}

function listAvailableTemplates() {
  try {
    var persistidos = wtgReadObjects_(REPORT_TEMPLATES_SHEET).map(function (t) { return String(t.ID); });
    var ids = {};
    Object.keys(REPORT_TEMPLATES_DEFAULT).forEach(function (k) { ids[k] = true; });
    persistidos.forEach(function (id) { ids[id] = true; });
    return Object.keys(ids);
  } catch (error) {
    Logger.log("Erro em listAvailableTemplates: " + error.message);
    throw error;
  }
}
