/**
 * AuthStandardWiring.gs — PILOTO de conexao do AuthStandardService (FROTA-17).
 *
 * Proposito: provar, em um projeto real da frota, que o servico padronizado de
 * autenticacao por token (AuthStandardService.gs) pode ser instanciado contra a
 * aba 'Usuarios' real e usado em producao, no mesmo padrao ja adotado por
 * "Como Crer", "Metateca.com" e "Tool Debate".
 *
 * O login web, o gateway e o roteador compartilham agora o mesmo caminho por
 * token (loginWithToken/isAuthenticatedByToken/logoutWithToken). As funções
 * legadas permanecem disponíveis para operações internas fora do web app, mas
 * não podem abrir uma sessão global para visitantes.
 *
 * Aceita senha em texto plano, conforme o contrato deste quiosque escolar.
 */

/** Resolve a planilha principal reutilizando o helper do projeto. */
function AuthStd_getSpreadsheet_() {
  try {
    if (typeof wtgGetSpreadsheet_ === 'function') {
      var ss = wtgGetSpreadsheet_();
      if (ss) return ss;
    }
  } catch (e) {}
  return getBoundSpreadsheet_();
}

/** Aceita apenas hashes SHA-256 hex (64 chars); caso contrario, vazio. */
function AuthStd_normalizarHash_(valor) {
  var v = String(valor == null ? '' : valor).trim();
  return /^[0-9a-fA-F]{64}$/.test(v) ? v : '';
}

/**
 * Adaptador findUser do AuthStandardService: localiza um usuario na aba
 * 'Usuarios' por username OU email (case-insensitive) e o devolve no formato
 * esperado pelo servico ({ id, username, role, active, password, passwordHash }).
 * @param {string} username
 * @return {?Object}
 */
function AuthStd_findUser_(username) {
  try {
    try {
      var ss = AuthStd_getSpreadsheet_();
      var sheet = ss.getSheetByName('Usuarios') ||
                  ss.getSheetByName('Users') ||
                  ss.getSheetByName('Usuários');
      if (!sheet || sheet.getLastRow() < 2) return null;
      var values = sheet.getDataRange().getValues();
      var h = values[0].map(function (x) { return String(x || '').trim().toLowerCase(); });
      var iUser = h.indexOf('username');
      var iPass = h.indexOf('password');
      var iHash = h.indexOf('passwordhash');
      var iRole = h.indexOf('role');
      var iNome = h.indexOf('nome');
      var iEmail = h.indexOf('email');
      var iId = h.indexOf('id');
      var iStatus = h.indexOf('status');
      if (iUser < 0 || (iPass < 0 && iHash < 0)) return null;

      var u = String(username || '').trim().toLowerCase();
      for (var r = 1; r < values.length; r++) {
        var row = values[r];
        var rowUser = String(row[iUser] || '').trim().toLowerCase();
        var rowEmail = iEmail >= 0 ? String(row[iEmail] || '').trim().toLowerCase() : '';
        if (rowUser !== u && rowEmail !== u) continue;
        return {
          id:           iId >= 0 && row[iId] ? row[iId] : rowUser,
          username:     row[iUser],
          name:         iNome >= 0 ? row[iNome] : row[iUser],
          email:        iEmail >= 0 ? row[iEmail] : '',
          role:         iRole >= 0 && row[iRole] ? row[iRole] : 'professor',
          active:       iStatus < 0 || String(row[iStatus]).trim().toLowerCase() !== 'inativo',
          password:     iPass >= 0 ? String(row[iPass] || '') : '',
          passwordHash: AuthStd_normalizarHash_(iHash >= 0 ? row[iHash] : '')
        };
      }
      return null;
    } catch (error) {
      Logger.log("Erro em AuthStd_findUser_: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em AuthStd_findUser_: " + error.message);
    throw error;
  }
}

/** Instancia o AuthStandardService configurado para este projeto. */
function AuthStd_service_() {
  return AuthStandardService.configure({
    sessionKey: 'WAYTOGO_AUTH_SESSION',
    sessionTtlSeconds: 21600,
    adapters: { findUser: AuthStd_findUser_ }
  });
}

// ---------------------------------------------------------------------------
// Caminho de sessao por TOKEN (ScriptProperties), compartilhado com
// AuthStandardService para que login, roteamento e logout leiam a mesma chave.
// ---------------------------------------------------------------------------

var AUTHSTD_TOK_PREFIX_ = 'WAYTOGO_AUTH_SESSION_TOK_';

/**
 * Valida credenciais via AuthStandardService e emite um token de sessao.
 * @return {{ success:boolean, token?:string, message?:string }}
 */
function loginWithToken(username, password) {
  try {
    try {
      if (!String(username || '').trim() || !String(password || '')) {
        return { success: false, message: 'Informe usuario e senha.' };
      }
      var result = AuthStd_service_().login(username, password);
      if (!result.ok) return { success: false, message: 'Credenciais invalidas.' };
      var token = String(result.token || '');
      if (!token) return { success: false, message: 'Erro ao criar sessao.' };
      var baseUrl = '';
      try { baseUrl = ScriptApp.getService().getUrl() || ''; } catch (ignoredUrl) {}
      return {
        success: true,
        token: token,
        user: result.user,
        redirectUrl: baseUrl ? baseUrl + '?page=app&tok=' + encodeURIComponent(token) : ''
      };
    } catch (error) {
      Logger.log("Erro em loginWithToken: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em loginWithToken: " + error.message);
    throw error;
  }
}

/** Indica se um token de sessao e valido e nao expirou. */
function isAuthenticatedByToken(tok) {
  try {
    if (typeof tok !== 'string' || tok.length < 1 || tok.length > 200) return false;
    return AuthStd_service_().isAuthenticatedByToken(tok);
  } catch (error) {
    Logger.log("Erro em isAuthenticatedByToken: " + error.message);
    throw error;
  }
}

/** Retorna a identidade vinculada ao token sem expor metadados internos. */
function getSessionUser(tok) {
  if (typeof tok !== 'string' || tok.length < 1 || tok.length > 200) return null;
  var props = PropertiesService.getScriptProperties();
  var raw = props.getProperty(AUTHSTD_TOK_PREFIX_ + tok);
  if (!raw) return null;
  try {
    var session = JSON.parse(raw);
    var expiresAt = Number(session.expiresAt);
    if (!session.userId || !isFinite(expiresAt) || expiresAt <= new Date().getTime()) {
      props.deleteProperty(AUTHSTD_TOK_PREFIX_ + tok);
      return null;
    }
    return {
      id: String(session.userId),
      username: String(session.username || session.userId),
      role: String(session.role || 'aluno').toLowerCase()
    };
  } catch (error) {
    props.deleteProperty(AUTHSTD_TOK_PREFIX_ + tok);
    return null;
  }
}

/** Revoga (logout) um token de sessao. */
function logoutWithToken(tok) {
  if (typeof tok !== 'string' || tok.length < 1 || tok.length > 200) return { ok: true };
  return AuthStd_service_().logout(tok);
}
