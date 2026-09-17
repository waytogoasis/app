// AuthUI.gs
//
// Funcionalidade Principal: Funções de Apps Script para interagir com as interfaces de autenticação.
//
// Descrição: Este script atua como uma ponte entre o frontend HTML de login/registro
//            e o backend `AuthService.gs`. Ele recebe requisições da UI, chama as funções
//            apropriadas do `AuthService.gs` e retorna os resultados para a interface.
//
// Integrações:
// - AuthService.gs: Para realizar operações de login e logout.
// - HtmlService.gs: Para servir as páginas `Login.html` e `Register.html`.
// - SessionManager.gs: Para gerenciar a sessão do usuário após o login.
//
// Funções Principais:
// - `processLoginRequest(username, password)`: Processa a requisição de login da UI.
// - `processRegisterRequest(userData)`: Processa a requisição de registro da UI.
// - `processLogoutRequest()`: Processa a requisição de logout da UI.
//
// Observações: Garante que as interações da interface do usuário com o backend sejam seguras e eficientes.

/**
 * Processa o login vindo de Login.html.
 * @return {Object} envelope { success, message, redirectUrl?, error? }
 */
function processLoginRequest(username, password) {
  try {
    if (!username || !password) {
      return apiFail('Usuário e senha são obrigatórios.', 'VALIDATION');
    }

    // O login web usa o mesmo token que o roteador valida. Não recrie uma
    // sessão global em ScriptProperties, pois ela seria compartilhada entre
    // visitantes quando o deployment executa como proprietário.
    if (typeof loginWithToken === 'function') {
      var tokenLogin = loginWithToken(username, password);
      if (!tokenLogin || !tokenLogin.success) {
        return apiFail((tokenLogin && tokenLogin.message) || 'Credenciais inválidas.', 'AUTH');
      }
      var tokenUser = tokenLogin.user || {};
      var tokenRes = apiOk({
        user: tokenUser,
        token: tokenLogin.token || ''
      }, 'Login efetuado com sucesso.');
      tokenRes.token = tokenLogin.token || '';
      tokenRes.redirectUrl = tokenLogin.redirectUrl || getPageUrl(
        roleHome_(String(tokenUser.role || 'aluno').toLowerCase()),
        tokenLogin.token
      );
      logActivitySafe_(tokenUser.id || tokenUser.userId || tokenUser.username, 'login');
      return tokenRes;
    }

    var user = doLogin(username, password);
    if (!user) {
      return apiFail('Credenciais inválidas.', 'AUTH');
    }
    createSession(user.id, user.role);
    logActivitySafe_(user.id, 'login');
    var res = apiOk({ user: { id: user.id, role: user.role } }, 'Login efetuado com sucesso.');
    res.redirectUrl = getPageUrl(roleHome_(user.role));
    return res;
  } catch (err) {
    return apiFail('Erro ao processar login: ' + err.message, 'EXCEPTION');
  }
}

/**
 * Processa o registro vindo de Register.html.
 * @return {Object} envelope
 */
function processRegisterRequest(userData) {
  try {
    if (!userData || !userData.username || !userData.password) {
      return apiFail('Dados de registro incompletos.', 'VALIDATION');
    }
    if (typeof createUser !== 'function') {
      return apiFail('Cadastro indisponível: camada de usuários não implementada.', 'NOT_IMPLEMENTED');
    }
    var created = createUser(userData);
    return apiOk({ user: created }, 'Cadastro realizado. Faça login para continuar.');
  } catch (err) {
    return apiFail('Erro ao processar registro: ' + err.message, 'EXCEPTION');
  }
}

/**
 * Processa o logout. Limpa a sessão e devolve a URL de redirecionamento.
 * @return {Object} envelope com redirectUrl para a página de Login
 */
function processLogoutRequest() {
  try {
    var session = getSession();
    if (session) logActivitySafe_(session.userId, 'logout');
  } catch (err) { /* best-effort */ }
  doLogout();
  var res = apiOk(null, 'Sessão encerrada.');
  res.redirectUrl = getPageUrl('Login');
  return res;
}

/** Registro de atividade best-effort (não quebra o fluxo se o logger for stub). */
function logActivitySafe_(userId, action) {
  try {
    if (typeof logActivity === 'function') logActivity(userId, action);
  } catch (err) { /* no-op */ }
}
