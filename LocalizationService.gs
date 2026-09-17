// LocalizationService.gs
//
// Funcionalidade Principal: Gerencia a localização e internacionalização do sistema.
//
// Descrição: Suporte a múltiplos idiomas com dicionários embutidos (pt-BR, en) sobreponíveis
//            pela aba `Translations`. Idioma atual persistido em cache do usuário.
//
// Integrações:
// - Google Planilha (aba `Translations`): traduções customizadas.
// - CacheManager.gs / CacheService: idioma da sessão.
//
// Funções Principais:
// - `getTranslation(key, lang)`: Retorna a tradução (planilha → dicionário → própria chave).
// - `setLanguage(lang)`: Define o idioma atual da sessão.
// - `getAvailableLanguages()`: Retorna a lista de idiomas suportados.

var LOCALIZATION_DEFAULT = {
  'pt-BR': { welcome: 'Bem-vindo', login: 'Entrar', logout: 'Sair', students: 'Alunos', reports: 'Relatórios', save: 'Salvar' },
  'en': { welcome: 'Welcome', login: 'Login', logout: 'Logout', students: 'Students', reports: 'Reports', save: 'Save' }
};
var LOCALIZATION_LANG_KEY = 'localization:lang';

function getAvailableLanguages() {
  try {
    return Object.keys(LOCALIZATION_DEFAULT);
  } catch (error) {
    Logger.log("Erro em getAvailableLanguages: " + error.message);
    throw error;
  }
}

function setLanguage(lang) {
  if (getAvailableLanguages().indexOf(lang) === -1) return { success: false, message: 'Idioma não suportado: ' + lang };
  try {
    if (typeof putInCache === 'function') putInCache(LOCALIZATION_LANG_KEY, lang, 1800);
    else CacheService.getUserCache().put(LOCALIZATION_LANG_KEY, lang, 1800);
  } catch (e) {}
  return { success: true, data: { lang: lang } };
}

function getCurrentLanguage() {
  try {
    var cached = (typeof getFromCache === 'function') ? getFromCache(LOCALIZATION_LANG_KEY) : CacheService.getUserCache().get(LOCALIZATION_LANG_KEY);
    if (cached) return cached;
  } catch (e) {}
  return 'pt-BR';
}

function getTranslation(key, lang) {
  lang = lang || getCurrentLanguage();
  // 1) Planilha Translations [Key, <lang>...] ou [Key, Lang, Value]
  try {
    var rows = wtgReadObjects_('Translations');
    var hit = rows.filter(function (r) { return String(r.Key) === String(key) && String(r.Lang || r.Language || '') === String(lang); })[0];
    if (hit && (hit.Value !== undefined)) return hit.Value;
  } catch (e) {}
  // 2) Dicionário embutido
  var dict = LOCALIZATION_DEFAULT[lang] || {};
  if (dict[key] !== undefined) return dict[key];
  // 3) Fallback: própria chave
  return key;
}
