// MenuService.gs
//
// Funcionalidade Principal: Cria menus personalizados na interface da Google Planilha.
//
// Descrição: Este script utiliza a API do Google Apps Script para adicionar itens de menu
//            personalizados à barra de menus da Google Planilha. Esses menus podem ser usados
//            para acionar funções específicas do Apps Script, como abrir sidebars, gerar relatórios
//            ou iniciar processos de simulação, facilitando a interação do usuário com o sistema.
//
// Integrações:
// - SpreadsheetApp (Apps Script): Interage com a interface da Google Planilha para criar menus.
// - SidebarService.gs: Pode ser usado para abrir sidebars a partir de itens de menu.
// - Outros Services: Funções de outros serviços podem ser chamadas pelos itens de menu.
//
// Funções Principais:
// - `onOpen()`: Função especial que é executada automaticamente quando a planilha é aberta,
//               criando o menu personalizado.
// - `createCustomMenu()`: Lógica para construir e adicionar o menu à interface.
//
// Observações: A função `onOpen()` é um gatilho simples do Apps Script e é essencial para
//              a inicialização do menu personalizado.

/**
 * Função especial executada automaticamente quando a planilha é aberta.
 * Cria o menu personalizado do sistema Way To Go.
 */
function onOpen() {
  createCustomMenu();
}

/**
 * Cria e adiciona menu personalizado à barra de menus da Google Planilha.
 * O menu é adaptado baseado no papel do usuário atual.
 */
function createCustomMenu() {
  try {
    var ui = SpreadsheetApp.getUi();
    var menu = ui.createMenu('Way To Go');
    
    // Obtém usuário atual para personalizar menu
    var currentUser = null;
    var isAdmin = false;
    var isProfessor = false;
    
    try {
      if (typeof getCurrentSessionUser === 'function') {
        currentUser = getCurrentSessionUser();
      } else if (typeof getSession === 'function') {
        var session = getSession();
        if (session && session.userId && typeof getUserById === 'function') {
          currentUser = getUserById(session.userId);
        }
      }
      
      if (currentUser) {
        var role = (currentUser.role || currentUser.Role || '').toLowerCase();
        isAdmin = role === 'admin' || role === 'administrator';
        isProfessor = role === 'professor' || role === 'teacher';
      }
    } catch (e) {
      Logger.log("Aviso: não foi possível obter usuário atual: " + e.message);
    }
    
    // Itens comuns para todos os usuários
    menu.addItem('📊 Dashboard', 'openDashboard_');
    menu.addSeparator();
    
    // Itens para professores e administradores
    if (isProfessor || isAdmin) {
      var managementSubmenu = ui.createMenu('👥 Gerenciamento');
      managementSubmenu.addItem('Gerenciar Alunos', 'openStudentManagement_');
      managementSubmenu.addItem('Gerenciar Turmas', 'openClassroomManagement_');
      managementSubmenu.addItem('Gerenciar Simulações', 'openSimulationManagement_');
      managementSubmenu.addItem('Gerenciar Pontuações', 'openScoreManagement_');
      menu.addSubMenu(managementSubmenu);
      menu.addSeparator();
      
      var reportSubmenu = ui.createMenu('📈 Relatórios');
      reportSubmenu.addItem('Relatório de Desempenho', 'generatePerformanceReport_');
      reportSubmenu.addItem('Relatório de Turma', 'generateClassroomReport_');
      reportSubmenu.addItem('Relatório Longitudinal', 'generateLongitudinalReport_');
      reportSubmenu.addItem('Exportar Dados (CSV)', 'exportDataCSV_');
      menu.addSubMenu(reportSubmenu);
      menu.addSeparator();
    }
    
    // Itens exclusivos para administradores
    if (isAdmin) {
      var adminSubmenu = ui.createMenu('⚙️ Administração');
      adminSubmenu.addItem('Gerenciar Usuários', 'openUserManagement_');
      adminSubmenu.addItem('Configurações do Sistema', 'openSystemSettings_');
      adminSubmenu.addItem('Criar Backup', 'createBackupPrompt_');
      adminSubmenu.addItem('Visualizar Logs de Auditoria', 'viewAuditLog_');
      adminSubmenu.addItem('Saúde do Sistema', 'showSystemHealth_');
      menu.addSubMenu(adminSubmenu);
      menu.addSeparator();
    }
    
    // Ferramentas e ajuda
    var helpSubmenu = ui.createMenu('❓ Ajuda');
    helpSubmenu.addItem('Sobre o Sistema', 'showAbout_');
    helpSubmenu.addItem('Documentação', 'openDocumentation_');
    helpSubmenu.addItem('Reportar Problema', 'reportIssue_');
    menu.addSubMenu(helpSubmenu);
    
    menu.addSeparator();
    menu.addItem('🔄 Atualizar Dados', 'refreshData_');
    
    menu.addToUi();
    
    Logger.log("Menu personalizado criado com sucesso");
  } catch (error) {
    Logger.log("Erro ao criar menu personalizado: " + error.message);
    // Não lança erro para não impedir abertura da planilha
  }
}

/**
 * Funções de callback do menu - abrem diferentes interfaces
 */

function openDashboard_() {
  try {
    var user = getCurrentUser_();
    var role = (user.role || user.Role || 'professor').toLowerCase();
    
    var dashboardMap = {
      'admin': 'DashboardAdmin',
      'administrator': 'DashboardAdmin',
      'professor': 'DashboardProfessor',
      'teacher': 'DashboardProfessor',
      'aluno': 'DashboardAluno',
      'student': 'DashboardAluno'
    };
    
    var page = dashboardMap[role] || 'DashboardProfessor';
    showSidebar(page, 'Dashboard - ' + (user.Nome || user.nome || user.Username || 'Usuário'));
  } catch (e) {
    showError_('Erro ao abrir dashboard: ' + e.message);
  }
}

function openStudentManagement_() {
  showSidebar('StudentManagement', 'Gerenciamento de Alunos');
}

function openClassroomManagement_() {
  showSidebar('ClassroomManagement', 'Gerenciamento de Turmas');
}

function openSimulationManagement_() {
  showSidebar('SimulationOverview', 'Gerenciamento de Simulações');
}

function openScoreManagement_() {
  showSidebar('ScoreManagement', 'Gerenciamento de Pontuações');
}

function openUserManagement_() {
  showSidebar('UserManagement', 'Gerenciamento de Usuários');
}

function openSystemSettings_() {
  showSidebar('Settings', 'Configurações do Sistema');
}

function generatePerformanceReport_() {
  try {
    SpreadsheetApp.getUi().alert('Gerando relatório de desempenho...\nEsta funcionalidade está em desenvolvimento.');
    // TODO: Implementar geração de relatório
  } catch (e) {
    showError_('Erro ao gerar relatório: ' + e.message);
  }
}

function generateClassroomReport_() {
  showSidebar('ClassroomReport', 'Relatório de Turma');
}

function generateLongitudinalReport_() {
  showSidebar('LongitudinalStudentReport', 'Relatório Longitudinal');
}

function exportDataCSV_() {
  try {
    if (typeof DataExportService !== 'undefined' && DataExportService.exportToCSV) {
      var result = DataExportService.exportToCSV();
      SpreadsheetApp.getUi().alert('Exportação concluída!\n' + (result.message || 'Dados exportados com sucesso.'));
    } else {
      SpreadsheetApp.getUi().alert('Funcionalidade de exportação não disponível no momento.');
    }
  } catch (e) {
    showError_('Erro ao exportar dados: ' + e.message);
  }
}

function createBackupPrompt_() {
  try {
    var ui = SpreadsheetApp.getUi();
    var response = ui.alert(
      'Criar Backup',
      'Deseja criar um backup completo da planilha?\n\nO backup será salvo no Google Drive.',
      ui.ButtonSet.YES_NO
    );
    
    if (response === ui.Button.YES) {
      var user = getCurrentUser_();
      if (typeof backupData_ === 'function') {
        var result = backupData_(user);
        if (result.success) {
          ui.alert('Backup criado com sucesso!\n\nNome: ' + result.backupName + '\n\nO arquivo está disponível no Google Drive.');
        } else {
          ui.alert('Erro ao criar backup: ' + (result.error || 'Erro desconhecido'));
        }
      } else {
        ui.alert('Funcionalidade de backup não disponível.');
      }
    }
  } catch (e) {
    showError_('Erro ao criar backup: ' + e.message);
  }
}

function viewAuditLog_() {
  showSidebar('UserActivityLogPage', 'Logs de Auditoria');
}

function showSystemHealth_() {
  showSidebar('SystemHealth', 'Saúde do Sistema');
}

function showAbout_() {
  try {
    var ui = SpreadsheetApp.getUi();
    ui.alert(
      'Way To Go - Sistema de Gestão Pedagógica',
      'Versão: 2.0\n\n' +
      'Sistema desenvolvido para gestão de simulações pedagógicas\n' +
      'e acompanhamento do desenvolvimento de alunos.\n\n' +
      '© 2024-2026 Escola Classe 115 Norte',
      ui.ButtonSet.OK
    );
  } catch (e) {
    Logger.log("Erro ao exibir sobre: " + e.message);
  }
}

function openDocumentation_() {
  try {
    var html = '<html><body><h2>Documentação</h2><p>Para mais informações, consulte os arquivos README.md e documentação de arquitetura no projeto.</p></body></html>';
    var ui = HtmlService.createHtmlOutput(html).setWidth(400).setHeight(300);
    SpreadsheetApp.getUi().showModalDialog(ui, 'Documentação');
  } catch (e) {
    showError_('Erro ao abrir documentação: ' + e.message);
  }
}

function reportIssue_() {
  try {
    var ui = SpreadsheetApp.getUi();
    ui.alert(
      'Reportar Problema',
      'Para reportar problemas, entre em contato com o administrador do sistema\n' +
      'ou utilize o sistema de Help Desk configurado.',
      ui.ButtonSet.OK
    );
  } catch (e) {
    Logger.log("Erro ao reportar problema: " + e.message);
  }
}

function refreshData_() {
  try {
    SpreadsheetApp.flush();
    SpreadsheetApp.getActiveSpreadsheet().toast('Dados atualizados!', 'Atualização', 3);
  } catch (e) {
    showError_('Erro ao atualizar dados: ' + e.message);
  }
}

/**
 * Funções auxiliares privadas
 */

function getCurrentUser_() {
  try {
    if (typeof getCurrentSessionUser === 'function') {
      return getCurrentSessionUser();
    } else if (typeof getSession === 'function') {
      var session = getSession();
      if (session && session.userId && typeof getUserById === 'function') {
        return getUserById(session.userId);
      }
    }
    return { id: 'unknown', role: 'professor', Nome: 'Usuário' };
  } catch (e) {
    return { id: 'unknown', role: 'professor', Nome: 'Usuário' };
  }
}

function showError_(message) {
  try {
    SpreadsheetApp.getUi().alert('Erro', message, SpreadsheetApp.getUi().ButtonSet.OK);
  } catch (e) {
    Logger.log("Erro: " + message);
  }
}
