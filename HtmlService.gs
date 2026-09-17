// HtmlService.gs
//
// Funcionalidade Principal: Serve arquivos HTML e injeta dados dinamicamente.
//
// Descrição: Este script é responsável por carregar templates HTML, injetar dados
//            (como informações do usuário logado ou dados de relatórios) e servir
//            as páginas web para o navegador. Ele atua como o controlador para as
//            interfaces de usuário.
//
// Integrações:
// - Todos os arquivos .html: Carrega e processa os templates HTML.
// - google.script.run: Permite a comunicação assíncrona entre o cliente (HTML) e o servidor (Apps Script).
// - SessionManager.gs: Para obter informações da sessão do usuário e injetá-las nas páginas.
//
// Funções Principais:
// - `serveHtml(templateName, data)`: Carrega um template HTML, injeta o roteador
//                                    de cliente e serve a página.
// - `include(filename)`: Função auxiliar para incluir outros arquivos HTML.
// - `getScriptUrl()` / `getPageUrl(page)`: URLs para navegação client-side.
//
// Observação: o ponto de entrada `doGet` foi consolidado em Main.gs (roteador
//             único). Este módulo cuida apenas da renderização das páginas.

/**
 * Carrega um template, injeta o roteador de cliente (navigateTo/logout/callServer)
 * e serve a página com <title> e viewport adicionados em tempo de execução.
 * @param {string} pageName  Nome do arquivo HTML (sem extensão).
 * @param {Object=} data     Dados opcionais expostos ao template como `data`.
 * @return {HtmlOutput}
 */
function serveHtml(pageName, data) {
  try {
    var template = HtmlService.createTemplateFromFile(pageName);
    template.data = data || {};
    var content = template.evaluate().getContent();
    content = injectClientRouter_(content);
    return HtmlService.createHtmlOutput(content)
      .setTitle('Sistema de Avaliação · ' + pageName)
      .addMetaTag('viewport', 'width=device-width, initial-scale=1')
      .setXFrameOptionsMode(HtmlService.XFrameOptionsMode.ALLOWALL);
  } catch (error) {
    Logger.log("Erro em serveHtml: " + error.message);
    throw error;
  }
}

/** Inclui o conteúdo de outro arquivo HTML (cabeçalho, rodapé, parciais). */
function include(filename) {
  try {
    return HtmlService.createHtmlOutputFromFile(filename).getContent();
  } catch (error) {
    Logger.log("Erro em include: " + error.message);
    throw error;
  }
}

/**
 * Compacta dados estáticos para uso em data URLs (como logos base64)
 * Remove todos os espaços em branco para otimizar o tamanho
 */
function includeInlineData(filename) {
  return HtmlService.createHtmlOutputFromFile(filename).getContent().replace(/\s+/g, '');
}


/** URL de uma página específica (navegação client-side por ?page=). */
function getPageUrl(page, authToken) {
  try {
    var base = ScriptApp.getService().getUrl() || '';
    var sep = base.indexOf('?') === -1 ? '?' : '&';
    var url = base + sep + 'page=' + encodeURIComponent(page);
    if (authToken) url += '&tok=' + encodeURIComponent(authToken);
    return url;
  } catch (error) {
    Logger.log("Erro em getPageUrl: " + error.message);
    throw error;
  }
}

/** Injeta o roteador de cliente antes de </body> (presente em toda página servida). */
function injectClientRouter_(content) {
  try {
    var router = include('GameAssets') + '\n' + include('ApiClient') + '\n' + include('ClientRouter');
    if (content.indexOf('</body>') !== -1) {
      return content.replace('</body>', router + '\n</body>');
    }
    return content + router;
  } catch (error) {
    Logger.log("Erro em injectClientRouter_: " + error.message);
    throw error;
  }
}
