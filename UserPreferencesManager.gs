// UserPreferencesManager.gs
//
// Funcionalidade Principal: Gerencia as preferências individuais dos usuários.
//
// Descrição: Permite que os usuários configurem preferências pessoais (idioma, tema, notificações).
//            As preferências são armazenadas e mescladas com os padrões do sistema.
//
// Integrações:
// - Google Planilha (aba `PreferenciasUsuarios`): Armazenamento das preferências.
// - UserService.gs (wtg* helpers): camada de CRUD reutilizável.
//
// Funções Principais:
// - `getPreferences(userId)`: Retorna as preferências (default + armazenadas) de um usuário.
// - `updatePreferences(userId, newPreferences)`: Atualiza (merge) as preferências de um usuário.
// - `getDefaultPreferences()`: Retorna as preferências padrão do sistema.

var PREFERENCIAS_SHEET = 'PreferenciasUsuarios';
var PREFERENCIAS_HEADERS = ['ID', 'UserID', 'Preferencias', 'CriadoEm', 'AtualizadoEm'];

function getDefaultPreferences() {
  return { idioma: 'pt-BR', tema: 'claro', notificacoes: true, tamanhoFonte: 'medio' };
}

function upm_findRaw_(userId) {
  try {
    return wtgReadObjects_(PREFERENCIAS_SHEET)
      .filter(function (p) { return String(p.UserID || p.userid || '') === String(userId); })[0] || null;
  } catch (error) {
    Logger.log("Erro em upm_findRaw_: " + error.message);
    throw error;
  }
}

function getPreferences(userId) {
  try {
    var defaults = getDefaultPreferences();
    var raw = upm_findRaw_(userId);
    if (!raw) return defaults;
    var stored; try { stored = JSON.parse(raw.Preferencias || '{}'); } catch (e) { stored = {}; }
    Object.keys(stored).forEach(function (k) { defaults[k] = stored[k]; });
    return defaults;
  } catch (error) {
    Logger.log("Erro em getPreferences: " + error.message);
    throw error;
  }
}

function updatePreferences(userId, newPreferences) {
  if (String(userId || '').trim() === '') return { success: false, message: 'userId obrigatorio.' };
  var raw = upm_findRaw_(userId);
  if (!raw) {
    return wtgCreateRecord_(PREFERENCIAS_SHEET, PREFERENCIAS_HEADERS, {
      UserID: userId, Preferencias: JSON.stringify(newPreferences || {})
    }, { required: ['UserID'] });
  }
  var atual; try { atual = JSON.parse(raw.Preferencias || '{}'); } catch (e) { atual = {}; }
  Object.keys(newPreferences || {}).forEach(function (k) { atual[k] = newPreferences[k]; });
  return wtgUpdateRecordById_(PREFERENCIAS_SHEET, raw.ID, { Preferencias: JSON.stringify(atual) });
}
