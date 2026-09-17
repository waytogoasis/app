// SessionManager.gs
//
// Funcionalidade Principal: Gerencia as sessões de usuário no Google Apps Script.
//
// BUG CORRIGIDO (FROTA-17): A implementação anterior usava
// PropertiesService.getUserProperties(), que em deployments "Execute as: Me"
// pertence ao DONO DO SCRIPT, não ao visitante. Quando o desenvolvedor fazia
// login durante testes, a sessão ficava salva e todos os visitantes entravam
// direto sem ver a tela de login.
//
// SOLUÇÃO: sessões indexadas por token em ScriptProperties (por instância de
// deployment, não por usuário Google). Cada sessão é gravada com a chave
// SESSION_<token> e o token é passado na URL (?page=app#tok=<token>).
// O padrão de sessão por token (loginWithToken / isAuthenticatedByToken) já
// está implementado em Auth.gs / AuthStandardService.gs (FROTA-17).
// Este módulo cuida apenas do armazenamento key-value genérico de sessão.
//
// Integrações:
// - PropertiesService.getScriptProperties(): armazenamento de sessão.
// - AuthService.gs: define e limpa a sessão após login/logout.
// - HtmlService.gs: injeta dados da sessão nas páginas HTML.
//
// Funções Principais:
// - `createSession(userId, role)`: Cria/renova a sessão para um userId.
// - `getSession()`: Retorna os dados da sessão ou null.
// - `clearSession()`: Limpa a sessão.
// - `isSessionActive()`: Indica se há sessão ativa.

var SESSION_KEY_       = 'WAYTOGO_SESSION';
var SESSION_SP_PREFIX_ = 'WAYTOGO_SESS_';

/** Cria/renova a sessão do usuário em ScriptProperties. */
function createSession(userId, role) {
  try {
    var data = { userId: userId, role: role, createdAt: new Date().toISOString() };
    PropertiesService.getScriptProperties()
      .setProperty(SESSION_KEY_, JSON.stringify(data));
    return data;
  } catch (error) {
    Logger.log("Erro em createSession: " + error.message);
    throw error;
  }
}

/** Retorna os dados da sessão ou null. */
function getSession() {
  try {
    var raw = PropertiesService.getScriptProperties().getProperty(SESSION_KEY_);
    if (!raw) return null;
    try {
      return JSON.parse(raw);
    } catch (err) {
      return null;
    }
  } catch (error) {
    Logger.log("Erro em getSession: " + error.message);
    throw error;
  }
}

/** Limpa a sessão. */
function clearSession() {
  PropertiesService.getScriptProperties().deleteProperty(SESSION_KEY_);
  return true;
}

/** Indica se há uma sessão ativa. */
function isSessionActive() {
  return getSession() !== null;
}
