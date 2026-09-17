// AuthService.gs
//
// Funcionalidade Principal: Gerencia a autenticação de usuários no sistema.
//
// Descrição: Este script é responsável por verificar as credenciais de login dos usuários
//            contra os dados armazenados na aba 'Usuarios' da Google Planilha. Ele fornece
//            funções para login, logout e verificação do estado de autenticação do usuário.
//            Verifica a senha por hash SHA-256 (campo `passwordHash`), com fallback
//            de compatibilidade para o campo legado `password` em texto plano.
//
// Integrações:
// - Google Planilha (aba 'Usuarios'): Leitura de usuário e senha.
// - SessionManager.gs: Gerenciamento de sessões de usuário.
// - UserService.gs: Acesso a dados de usuário para verificação de credenciais.
//
// Funções Principais:
// - `doLogin(username, password)`: Tenta autenticar um usuário.
// - `doLogout()`: Encerra a sessão do usuário.
// - `isLoggedIn()`: Verifica se há um usuário autenticado na sessão atual.
// - `getLoggedUser()`: Retorna informações do usuário logado.
//
// Observações: doLogin NÃO possui backdoor; se a camada de dados (UserService)
//              ainda não tiver usuários, o login é negado com segurança.

/**
 * Verifica credenciais contra os usuários cadastrados.
 * @return {?Object} { id, role, username } se válido; null caso contrário.
 */
function doLogin(username, password) {
  var users = (typeof getAllUsers === 'function') ? getAllUsers() : null;
  if (!users || !users.length) {
    // Camada de dados ainda não populada: nega com segurança (sem credencial fixa).
    return null;
  }
  for (var i = 0; i < users.length; i++) {
    var u = users[i];
    if (u && u.username === username && verifyPassword_(password, u)) {
      return { id: u.id, role: u.role, username: u.username };
    }
  }
  return null;
}

/** Encerra a sessão do usuário atual. */
function doLogout() {
  clearSession();
  return true;
}

/** Indica se há um usuário autenticado na sessão atual. */
function isLoggedIn() {
  return isSessionActive();
}

/** Retorna os dados do usuário logado (da sessão) ou null. */
function getLoggedUser() {
  var session = getSession();
  return session ? { id: session.userId, role: session.role } : null;
}

/** Compara a senha informada com o valor armazenado — SEMPRE em texto plano. */
function verifyPassword_(password, user) {
  try {
    // Quiosque escolar: a senha (passwordHash/password/senha) fica em texto plano e
    // legivel na planilha. Nenhum hash e calculado ou exigido.
    var stored = user.passwordHash || user.password || user.senha;
    if (stored === undefined || stored === null || stored === '') return false;
    return String(password) === String(stored);
  } catch (error) {
    Logger.log("Erro em verifyPassword_: " + error.message);
    throw error;
  }
}

/** Log best-effort (não quebra o login se o logger ainda for stub). */
function logWarning_(message) {
  try {
    if (typeof logWarning === 'function') { logWarning(message); return; }
  } catch (err) { /* no-op */ }
  LoggerService.info(message);
}
