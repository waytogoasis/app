// DashboardRenderer.gs
//
// Funcionalidade Principal: Renderiza os dashboards dinamicamente (HTML).
//
// Descrição: Combina dados (DashboardData) e configurações de gráfico (ChartGenerator) em
//            fragmentos HTML para os dashboards. A montagem do HTML é determinística e testável;
//            a entrega ao navegador fica a cargo do HtmlService no ambiente GAS.
//
// Integrações:
// - DashboardData.gs / ChartGenerator.gs.
//
// Funções Principais:
// - `renderAdminDashboard(data)`: HTML do dashboard do administrador.
// - `renderProfessorDashboard(data)`: HTML do dashboard do professor.
// - `renderAlunoDashboard(data)`: HTML do dashboard do aluno.

function dr_esc_(v) {
  try {
    return String(v === null || v === undefined ? '' : v)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  } catch (error) {
    Logger.log("Erro em dr_esc_: " + error.message);
    throw error;
  }
}

function dr_card_(titulo, valor) {
  return '<div class="card"><h3>' + dr_esc_(titulo) + '</h3><p class="valor">' + dr_esc_(valor) + '</p></div>';
}

function renderAdminDashboard(data) {
  try {
    data = data || ((typeof getAdminDashboardData === 'function') ? getAdminDashboardData() : {});
    var stats = data.stats || {};
    var html = '<section class="dashboard admin"><h1>Painel do Administrador</h1>';
    html += dr_card_('Alunos', stats.totalAlunos || 0);
    html += dr_card_('Simulações', stats.totalSimulacoes || 0);
    html += dr_card_('Média Geral', stats.mediaGeral || 0);
    html += dr_card_('Turmas', data.totalTurmas || 0);
    html += '<ul class="top-turmas">';
    (data.topTurmas || []).forEach(function (t) {
      html += '<li>' + dr_esc_(t.nome || t.classId) + ': ' + dr_esc_(t.media) + '</li>';
    });
    html += '</ul></section>';
    return html;
  } catch (error) {
    Logger.log("Erro em renderAdminDashboard: " + error.message);
    throw error;
  }
}

function renderProfessorDashboard(data) {
  try {
    var html = '<section class="dashboard professor"><h1>Painel do Professor</h1>';
    html += dr_card_('Turmas', (data && data.totalTurmas) || 0);
    html += '<ul class="turmas">';
    ((data && data.turmas) || []).forEach(function (t) {
      html += '<li>' + dr_esc_(t.nome) + ' — ' + dr_esc_(t.alunos) + ' alunos, média ' + dr_esc_(t.mediaTurma) + '</li>';
    });
    html += '</ul></section>';
    return html;
  } catch (error) {
    Logger.log("Erro em renderProfessorDashboard: " + error.message);
    throw error;
  }
}

function renderAlunoDashboard(data) {
  var m = (data && data.metricas) || {};
  var p = (data && data.progresso) || {};
  var html = '<section class="dashboard aluno"><h1>Meu Painel</h1>';
  html += dr_card_('Média', m.mediaGeral || p.media || 0);
  html += dr_card_('Simulações', m.simulacoes || 0);
  html += dr_card_('Conquistas', (data && data.conquistas) || 0);
  html += dr_card_('Evolução', p.evolucao || 0);
  html += '</section>';
  return html;
}
