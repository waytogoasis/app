// ConfigService.gs
//
// Funcionalidade Principal: Gerencia as configurações globais do sistema.
//
// Descrição: Este script permite a leitura e atualização de parâmetros de configuração
//            do sistema, armazenados na aba `Settings` da Google Planilha e/ou como
//            propriedades de script do Apps Script (fallback para chaves estáticas/sensíveis).
//
// Integrações:
// - Google Planilha (aba `Settings`): Armazenamento de configurações dinâmicas.
// - PropertiesService (Apps Script): Armazenamento de configurações estáticas ou sensíveis.
// - SpreadsheetUtils.gs: Utiliza funções auxiliares para manipulação da planilha de configurações.
// - Todos os Services: Podem acessar configurações via este serviço.
//
// Funções Principais:
// - `getSetting(key)`: Retorna o valor de uma configuração (planilha → Script Properties).
// - `setSetting(key, value)`: Define ou atualiza o valor de uma configuração na planilha.
// - `getAllSettings()`: Retorna todas as configurações da planilha como objeto.
//
// Observações: Configurações dinâmicas residem na aba Settings; segredos em Script Properties.

var CONFIG_SHEET = 'Settings';

function getSetting(key) {
  // 1) Aba Settings (dinâmica)
  try {
    var rows = getDataAsObjects(CONFIG_SHEET);
    for (var i = 0; i < rows.length; i++) {
      if (String(rows[i].Key) === String(key)) return rows[i].Value;
    }
  } catch (e) {}
  // 2) Script Properties (estática/sensível)
  try {
    var prop = PropertiesService.getScriptProperties().getProperty(key);
    if (prop !== null && prop !== undefined) return prop;
  } catch (e2) {}
  return null;
}

function setSetting(key, value, description) {
  var now = new Date().toISOString();
  var found = null;
  try { found = findRow(CONFIG_SHEET, 0, key); } catch (e) {}
  if (found) {
    // Estrutura: [Key, Value, Description, Scope, UpdatedAt, UpdatedBy]
    var row = found.row.slice();
    row[1] = value;
    row[4] = now;
    return updateRow(CONFIG_SHEET, found.rowIndex, row);
  }
  return appendRow(CONFIG_SHEET, [key, value, description || '', 'app', now, 'system']);
}

function getAllSettings() {
  var out = {};
  try {
    getDataAsObjects(CONFIG_SHEET).forEach(function(r) {
      if (r.Key !== undefined && r.Key !== '') out[r.Key] = r.Value;
    });
  } catch (e) {}
  return out;
}
