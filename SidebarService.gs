// SidebarService.gs
//
// Funcionalidade Principal: Gerencia a exibição de sidebars (barras laterais) em editores do Google (Sheets, Docs, etc.).
//
// Descrição: Este script permite criar e exibir interfaces de usuário personalizadas em sidebars
//            dentro dos aplicativos do Google Workspace. É útil para fornecer ferramentas
//            adicionais ou informações contextuais aos usuários enquanto trabalham na planilha.
//
// Integrações:
// - HtmlService.gs: Utiliza para carregar e exibir conteúdo HTML dentro da sidebar.
// - Google Workspace UI (SpreadsheetApp, DocumentApp, etc.): Interage com a interface do usuário.
//
// Funções Principais:
// - `showSidebar(htmlTemplateName, title)`: Exibe uma sidebar com o conteúdo HTML especificado.
// - `closeSidebar()`: Fecha a sidebar atualmente aberta.
//
// Observações: A funcionalidade de sidebar é específica para o ambiente de editores do Google.

/**
 * Exibe uma sidebar com conteúdo HTML especificado.
 * @param {string} htmlTemplateName - Nome do arquivo HTML a ser exibido (sem extensão .html)
 * @param {string} [title] - Título da sidebar (opcional, padrão: nome do template)
 * @param {Object} [data] - Dados adicionais para passar ao template (opcional)
 * @return {HtmlOutput} Objeto HtmlOutput exibido na sidebar
 */
function showSidebar(htmlTemplateName, title, data) {
  try {
    if (!htmlTemplateName) {
      throw new Error("Nome do template HTML é obrigatório");
    }
    
    // Remove extensão .html se fornecida
    var templateName = htmlTemplateName.replace(/\.html$/i, '');
    
    // Define título padrão se não fornecido
    var sidebarTitle = title || templateName;
    
    // Cria o HtmlOutput a partir do template
    var htmlOutput;
    
    try {
      // Tenta criar template do HtmlService
      var template = HtmlService.createTemplateFromFile(templateName);
      
      // Injeta dados no template se fornecidos
      if (data && typeof data === 'object') {
        for (var key in data) {
          if (data.hasOwnProperty(key)) {
            template[key] = data[key];
          }
        }
      }
      
      // Tenta injetar informações do usuário atual
      try {
        if (typeof getCurrentSessionUser === 'function') {
          template.currentUser = getCurrentSessionUser();
        } else if (typeof getSession === 'function') {
          var session = getSession();
          if (session && session.userId && typeof getUserById === 'function') {
            template.currentUser = getUserById(session.userId);
          }
        }
      } catch (userError) {
        Logger.log("Aviso: não foi possível injetar usuário no template: " + userError.message);
      }
      
      // Avalia o template
      htmlOutput = template.evaluate()
        .setTitle(sidebarTitle)
        .setWidth(400);
      
    } catch (templateError) {
      // Fallback: tenta criar HTML direto do arquivo
      Logger.log("Erro ao criar template, tentando HTML direto: " + templateError.message);
      htmlOutput = HtmlService.createHtmlOutputFromFile(templateName)
        .setTitle(sidebarTitle)
        .setWidth(400);
    }
    
    // Configura sandbox mode para segurança
    htmlOutput.setSandboxMode(HtmlService.SandboxMode.IFRAME);
    
    // Exibe a sidebar
    SpreadsheetApp.getUi().showSidebar(htmlOutput);
    
    Logger.log("Sidebar exibida com sucesso: " + templateName);
    
    return htmlOutput;
  } catch (error) {
    Logger.log("Erro em showSidebar: " + error.message);
    
    // Tenta exibir erro na sidebar
    try {
      var errorHtml = HtmlService.createHtmlOutput(
        '<div style="padding: 20px; font-family: Arial, sans-serif;">' +
        '<h2 style="color: #d32f2f;">Erro ao Carregar Sidebar</h2>' +
        '<p><strong>Template:</strong> ' + htmlTemplateName + '</p>' +
        '<p><strong>Erro:</strong> ' + error.message + '</p>' +
        '<p style="margin-top: 20px; font-size: 12px; color: #666;">' +
        'Verifique se o arquivo HTML existe e está corretamente configurado.' +
        '</p>' +
        '</div>'
      )
        .setTitle('Erro')
        .setWidth(400);
      
      SpreadsheetApp.getUi().showSidebar(errorHtml);
    } catch (errorDisplayError) {
      // Se nem o erro pode ser exibido, mostra alert
      SpreadsheetApp.getUi().alert('Erro ao exibir sidebar: ' + error.message);
    }
    
    throw error;
  }
}

/**
 * Fecha a sidebar atualmente aberta.
 * Nota: O Google Apps Script não possui método direto para fechar sidebar,
 * então esta função exibe uma sidebar vazia ou pequena mensagem.
 * @return {boolean} true se operação foi bem-sucedida
 */
function closeSidebar() {
  try {
    // Apps Script não tem método nativo para fechar sidebar
    // Workaround: exibe uma sidebar mínima com mensagem de fechamento
    var closedHtml = HtmlService.createHtmlOutput(
      '<div style="padding: 20px; font-family: Arial, sans-serif; text-align: center;">' +
      '<p style="color: #666;">Sidebar fechada</p>' +
      '<p style="font-size: 12px; margin-top: 10px;">' +
      '<em>Você pode fechar esta barra lateral manualmente clicando no X</em>' +
      '</p>' +
      '</div>'
    )
      .setTitle('Fechado')
      .setWidth(250);
    
    SpreadsheetApp.getUi().showSidebar(closedHtml);
    
    Logger.log("Sidebar 'fechada' (exibindo mensagem de fechamento)");
    
    return true;
  } catch (error) {
    Logger.log("Erro em closeSidebar: " + error.message);
    return false;
  }
}

/**
 * Exibe um modal dialog ao invés de sidebar (alternativa).
 * @param {string} htmlTemplateName - Nome do arquivo HTML
 * @param {string} [title] - Título do modal
 * @param {number} [width] - Largura do modal (padrão: 600)
 * @param {number} [height] - Altura do modal (padrão: 400)
 * @param {Object} [data] - Dados para passar ao template
 */
function showModal(htmlTemplateName, title, width, height, data) {
  try {
    if (!htmlTemplateName) {
      throw new Error("Nome do template HTML é obrigatório");
    }
    
    var templateName = htmlTemplateName.replace(/\.html$/i, '');
    var modalTitle = title || templateName;
    var modalWidth = width || 600;
    var modalHeight = height || 400;
    
    var template = HtmlService.createTemplateFromFile(templateName);
    
    // Injeta dados
    if (data && typeof data === 'object') {
      for (var key in data) {
        if (data.hasOwnProperty(key)) {
          template[key] = data[key];
        }
      }
    }
    
    // Injeta usuário atual
    try {
      if (typeof getCurrentSessionUser === 'function') {
        template.currentUser = getCurrentSessionUser();
      }
    } catch (e) {}
    
    var htmlOutput = template.evaluate()
      .setWidth(modalWidth)
      .setHeight(modalHeight);
    
    SpreadsheetApp.getUi().showModalDialog(htmlOutput, modalTitle);
    
    Logger.log("Modal exibido com sucesso: " + templateName);
  } catch (error) {
    Logger.log("Erro em showModal: " + error.message);
    SpreadsheetApp.getUi().alert('Erro ao exibir modal: ' + error.message);
    throw error;
  }
}

/**
 * Exibe uma sidebar modeless (não-modal, permite interação com a planilha).
 * @param {string} htmlTemplateName - Nome do arquivo HTML
 * @param {string} [title] - Título
 * @param {Object} [data] - Dados para o template
 */
function showModelessDialog(htmlTemplateName, title, data) {
  try {
    if (!htmlTemplateName) {
      throw new Error("Nome do template HTML é obrigatório");
    }
    
    var templateName = htmlTemplateName.replace(/\.html$/i, '');
    var dialogTitle = title || templateName;
    
    var template = HtmlService.createTemplateFromFile(templateName);
    
    if (data && typeof data === 'object') {
      for (var key in data) {
        if (data.hasOwnProperty(key)) {
          template[key] = data[key];
        }
      }
    }
    
    var htmlOutput = template.evaluate()
      .setWidth(500)
      .setHeight(400);
    
    SpreadsheetApp.getUi().showModelessDialog(htmlOutput, dialogTitle);
    
    Logger.log("Modeless dialog exibido com sucesso: " + templateName);
  } catch (error) {
    Logger.log("Erro em showModelessDialog: " + error.message);
    SpreadsheetApp.getUi().alert('Erro ao exibir dialog: ' + error.message);
    throw error;
  }
}

/**
 * Função auxiliar para incluir arquivos HTML (para uso em templates).
 * Permite modularização de HTML com includes.
 * @param {string} filename - Nome do arquivo a incluir
 * @return {string} Conteúdo do arquivo HTML
 */
function sidebarInclude_(filename) {
  try {
    return HtmlService.createHtmlOutputFromFile(filename).getContent();
  } catch (error) {
    Logger.log("Erro ao incluir arquivo " + filename + ": " + error.message);
    return '<!-- Erro ao incluir: ' + filename + ' -->';
  }
}
