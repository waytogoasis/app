// Main.gs
//
// Funcionalidade Principal: Ponto de entrada principal para o projeto Google Apps Script.
//
// Descrição: Este script contém as funções `doGet` e `doPost`, que são os manipuladores
//            de requisições HTTP para aplicações web publicadas no Apps Script. Ele atua
//            como o roteador inicial, direcionando as requisições para os serviços apropriados
//            (ex: autenticação, servir páginas HTML, processar dados).
//
// Integrações:
// - HtmlService.gs: Utilizado para servir as páginas HTML da aplicação.
// - AuthService.gs: Pode ser chamado para verificar o estado de autenticação do usuário.
// - ApiEndpoints.gs: Pode rotear requisições POST/GET para endpoints de API.
// - MenuService.gs: A função `onOpen()` pode ser definida aqui para criar menus personalizados.
//
// Funções Principais:
// - `doGet(e)`: ROTEADOR ÚNICO. Resolve a página por `?page=`, aplica o gate de
//               sessão/papel e delega a renderização para HtmlService.serveHtml.
// - `doPost(e)`: Ponto de entrada POST único; delega para a camada de API.
// - `roleHome_(role)`: Mapeia o papel do usuário para seu dashboard inicial.
//
// Observação: ESTE é o único `doGet`/`doPost` do projeto. As versões antigas em
//             HtmlService.gs e ApiEndpoints.gs foram removidas/renomeadas para
//             evitar conflito de nomes reservados no Apps Script.

// Páginas acessíveis sem login.
var PUBLIC_PAGES = ['Login', 'Register', 'ErrorPage'];

// Corte vertical publicado enquanto as telas legadas migram para apiCall.
// Manter uma allowlist é preferível a expor páginas que ainda chamam globals
// inexistentes e só exibem erro depois do clique.
var RELEASE_PAGES = ['Login', 'Register', 'ErrorPage', 'DashboardAdmin', 'DashboardProfessor', 'DashboardAluno'];

function doGet(e) {
  try {
    var params = e && e.parameter ? e.parameter : {};
    var tok = params.tok || '';

    if (params.api === '1' && typeof handleApiGet === 'function') {
      return handleApiGet(e);
    }

    if (params.page === 'features') {
      return serveHtml('ErrorPage', { message: 'Esta tela está em preparação até a integração com o backend ser homologada.' });
    }

    var requested = params.page === 'app' ? null : (params.page || null);
    var tokenValido = tok && typeof isAuthenticatedByToken === 'function' && isAuthenticatedByToken(tok);
    var session = null;
    var loggedIn = false;

    if (tokenValido) {
      var tokenUser = getSessionUser(tok) || {};
      session = {
        userId: tokenUser.id || tokenUser.userId || tokenUser.username || 'usuario',
        username: tokenUser.username || 'usuario',
        role: String(tokenUser.role || 'aluno').toLowerCase(),
        authToken: tok
      };
      loggedIn = true;
    }

    if (!loggedIn) {
      var target = (requested && PUBLIC_PAGES.indexOf(requested) !== -1) ? requested : 'Login';
      return serveHtml(target);
    }

    if (!requested || PUBLIC_PAGES.indexOf(requested) !== -1) {
      requested = roleHome_(session.role);
    }

    if (RELEASE_PAGES.indexOf(requested) === -1) {
      Logger.log('Página fora da superfície homologada: ' + requested);
      return serveHtml('ErrorPage', {
        message: 'Esta tela está em preparação até a integração com o backend ser homologada.'
      });
    }
    return serveHtml(requested, { session: session, authToken: tok });
  } catch (error) {
    Logger.log("Erro em doGet: " + error.message);
    throw error;
  }
}


function doPost(e) {
  try {
    try {
      if (typeof handleApiPost === 'function') {
        return handleApiPost(e);
      }
      return ContentService
        .createTextOutput(JSON.stringify(apiFail('Nenhum handler POST configurado.', 'NOT_IMPLEMENTED')))
        .setMimeType(ContentService.MimeType.JSON);
    } catch (error) {
      Logger.log("Erro em doPost: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em doPost: " + error.message);
    throw error;
  }
}

/** Dashboard inicial por papel do usuário. */
function roleHome_(role) {
  switch (role) {
    case 'admin':     return 'DashboardAdmin';
    case 'professor': return 'DashboardProfessor';
    case 'aluno':     return 'DashboardAluno';
    default:          return 'DashboardAluno';
  }
}
