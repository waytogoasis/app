// DashboardUI.gs
//
// Funcionalidade Principal: Funções de Apps Script para interagir com as interfaces de dashboard.
//
// Descrição: Este script atua como uma ponte entre os dashboards HTML (Admin, Professor, Aluno)
//            e o backend `DashboardData.gs` e `ChartGenerator.gs`. Ele recebe requisições da UI,
//            chama as funções apropriadas para obter os dados e retorna-os para a interface
//            para renderização.
//
// Integrações:
// - DashboardData.gs: Para obter os dados brutos do dashboard.
// - ChartGenerator.gs: Para obter os dados formatados para gráficos.
// - HtmlService.gs: Para servir as páginas HTML dos dashboards.
// - SessionManager.gs: Para identificar o tipo de usuário logado e servir o dashboard correto.
//
// Funções Principais:
// - `getDashboardDataForUser()`: Retorna os dados do dashboard apropriado para o usuário logado.
// - `getChartDataForDashboard(chartType, alunoId)`: Retorna dados de gráfico para o dashboard.
//
// Observações: Centraliza a lógica de carregamento de dados para os diferentes dashboards.

/**
 * Retorna os dados do dashboard apropriados para o usuário logado.
 * Agrega informações de alunos, simulações, pontuações e estatísticas.
 * @param {Object} [user] - Usuário (opcional, usa sessão atual se não fornecido)
 * @return {Object} Dados do dashboard formatados por tipo de usuário
 */
function getDashboardDataForUser_(user) {
  try {
    // Obtém usuário atual se não fornecido
    if (!user) {
      try {
        if (typeof getCurrentSessionUser === 'function') {
          user = getCurrentSessionUser();
        } else if (typeof getSession === 'function') {
          var session = getSession();
          if (session && session.userId && typeof getUserById === 'function') {
            user = getUserById(session.userId);
          }
        }
      } catch (e) {
        Logger.log("Erro ao obter usuário da sessão: " + e.message);
      }
    }

    if (!user || !user.role) {
      return {
        success: false,
        error: 'Usuário não identificado ou sem papel definido'
      };
    }

    var role = String(user.role || user.Role || '').toLowerCase();
    var dashboardData = {
      success: true,
      user: {
        id: user.ID || user.id,
        nome: user.Nome || user.nome || user.Username || 'Usuário',
        role: role
      },
      timestamp: new Date().toISOString()
    };

    // Dados específicos por papel
    if (role === 'admin' || role === 'administrator') {
      dashboardData.data = getDashboardDataAdmin_();
    } else if (role === 'professor' || role === 'teacher') {
      dashboardData.data = getDashboardDataProfessor_(user);
    } else if (role === 'aluno' || role === 'student') {
      dashboardData.data = getDashboardDataAluno_(user);
    } else {
      dashboardData.data = getDashboardDataProfessor_(user); // fallback
    }

    return dashboardData;
  } catch (error) {
    Logger.log("Erro em getDashboardDataForUser: " + error.message);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Retorna dados de gráfico formatados para o dashboard.
 * @param {string} chartType - Tipo de gráfico (performance, attendance, progress, etc.)
 * @param {string|number} [alunoId] - ID do aluno (para gráficos específicos)
 * @param {Object} [options] - Opções adicionais (período, filtros, etc.)
 * @return {Object} Dados formatados para o gráfico
 */
function getChartDataForDashboard_(chartType, alunoId, options) {
  try {
    options = options || {};
    
    if (!chartType) {
      return { success: false, error: 'Tipo de gráfico não especificado' };
    }

    var chartData = {
      success: true,
      chartType: chartType,
      timestamp: new Date().toISOString()
    };

    // Roteamento por tipo de gráfico
    switch (chartType.toLowerCase()) {
      case 'performance':
      case 'desempenho':
        chartData.data = getPerformanceChartData_(alunoId, options);
        break;
      
      case 'attendance':
      case 'frequencia':
        chartData.data = getAttendanceChartData_(options);
        break;
      
      case 'progress':
      case 'progresso':
        chartData.data = getProgressChartData_(alunoId, options);
        break;
      
      case 'comparison':
      case 'comparacao':
        chartData.data = getComparisonChartData_(options);
        break;
      
      case 'distribution':
      case 'distribuicao':
        chartData.data = getDistributionChartData_(options);
        break;
      
      default:
        return { success: false, error: 'Tipo de gráfico não suportado: ' + chartType };
    }

    return chartData;
  } catch (error) {
    Logger.log("Erro em getChartDataForDashboard: " + error.message);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Funções privadas para obter dados específicos por papel
 */

function getDashboardDataAdmin_() {
  var data = {
    statistics: {},
    recentActivity: [],
    systemHealth: {}
  };

  try {
    // Estatísticas gerais do sistema
    var totalUsers = 0, totalAlunos = 0, totalSimulacoes = 0, totalPontuacoes = 0;
    
    try {
      if (typeof getAllUsers === 'function') {
        totalUsers = getAllUsers().length;
      }
    } catch (e) { Logger.log("Erro ao contar usuários: " + e.message); }
    
    try {
      if (typeof getAllAlunos === 'function') {
        totalAlunos = getAllAlunos().length;
      }
    } catch (e) { Logger.log("Erro ao contar alunos: " + e.message); }
    
    try {
      if (typeof getAllSimulations === 'function') {
        totalSimulacoes = getAllSimulations().length;
      }
    } catch (e) { Logger.log("Erro ao contar simulações: " + e.message); }
    
    try {
      if (typeof wtgReadObjects_ === 'function') {
        totalPontuacoes = wtgReadObjects_('Pontuacoes').length;
      }
    } catch (e) { Logger.log("Erro ao contar pontuações: " + e.message); }

    data.statistics = {
      totalUsers: totalUsers,
      totalAlunos: totalAlunos,
      totalSimulacoes: totalSimulacoes,
      totalPontuacoes: totalPontuacoes
    };

    // Atividades recentes do log de auditoria
    try {
      if (typeof getAuditLog === 'function') {
        data.recentActivity = getAuditLog({ limit: 10 });
      }
    } catch (e) { Logger.log("Erro ao obter atividades recentes: " + e.message); }

    // Saúde do sistema
    data.systemHealth = {
      status: 'operational',
      lastBackup: 'N/A',
      uptime: '99.9%'
    };

  } catch (e) {
    Logger.log("Erro em getDashboardDataAdmin_: " + e.message);
  }

  return data;
}

function getDashboardDataProfessor_(user) {
  var data = {
    statistics: {},
    recentSimulations: [],
    studentSummary: []
  };

  try {
    // Estatísticas do professor
    var alunos = [];
    try {
      if (typeof getAllAlunos === 'function') {
        alunos = getAllAlunos();
      }
    } catch (e) { Logger.log("Erro ao obter alunos: " + e.message); }

    var simulacoes = [];
    try {
      if (typeof getAllSimulations === 'function') {
        simulacoes = getAllSimulations();
      }
    } catch (e) { Logger.log("Erro ao obter simulações: " + e.message); }

    data.statistics = {
      totalAlunos: alunos.length,
      totalSimulacoes: simulacoes.length,
      simulacoesAtivas: simulacoes.filter(function(s) { 
        return (s.Status || s.status || '').toLowerCase() === 'em_andamento'; 
      }).length
    };

    // Simulações recentes (últimas 5)
    data.recentSimulations = simulacoes
      .sort(function(a, b) {
        var dateA = new Date(a.CriadoEm || a.criadoEm || 0);
        var dateB = new Date(b.CriadoEm || b.criadoEm || 0);
        return dateB - dateA;
      })
      .slice(0, 5);

    // Resumo de alunos com pontuações médias
    data.studentSummary = alunos.slice(0, 10).map(function(aluno) {
      var pontuacoes = [];
      try {
        if (typeof getPontuacoesByAluno === 'function') {
          pontuacoes = getPontuacoesByAluno(aluno.ID || aluno.id);
        }
      } catch (e) {}

      var totalScore = 0;
      if (pontuacoes.length > 0) {
        totalScore = pontuacoes.reduce(function(sum, p) {
          return sum + (Number(p.Total || p.total) || 0);
        }, 0) / pontuacoes.length;
      }

      return {
        id: aluno.ID || aluno.id,
        nome: aluno.Nome || aluno.nome,
        pontuacaoMedia: Math.round(totalScore * 100) / 100,
        totalSimulacoes: pontuacoes.length
      };
    });

  } catch (e) {
    Logger.log("Erro em getDashboardDataProfessor_: " + e.message);
  }

  return data;
}

function getDashboardDataAluno_(user) {
  var data = {
    statistics: {},
    mySimulations: [],
    myScores: [],
    progress: {}
  };

  try {
    var alunoId = user.ID || user.id;

    // Minhas simulações
    try {
      if (typeof getSimulationsByAluno === 'function') {
        data.mySimulations = getSimulationsByAluno(alunoId);
      }
    } catch (e) { Logger.log("Erro ao obter simulações do aluno: " + e.message); }

    // Minhas pontuações
    try {
      if (typeof getPontuacoesByAluno === 'function') {
        data.myScores = getPontuacoesByAluno(alunoId);
      }
    } catch (e) { Logger.log("Erro ao obter pontuações do aluno: " + e.message); }

    // Estatísticas pessoais
    var totalScore = 0;
    if (data.myScores.length > 0) {
      totalScore = data.myScores.reduce(function(sum, p) {
        return sum + (Number(p.Total || p.total) || 0);
      }, 0) / data.myScores.length;
    }

    data.statistics = {
      totalSimulacoes: data.mySimulations.length,
      pontuacaoMedia: Math.round(totalScore * 100) / 100,
      ultimaSimulacao: data.mySimulations.length > 0 
        ? data.mySimulations[data.mySimulations.length - 1].CriadoEm 
        : 'N/A'
    };

    // Progresso ao longo do tempo
    data.progress = {
      labels: data.myScores.slice(-5).map(function(s, i) { return 'Sim ' + (i + 1); }),
      values: data.myScores.slice(-5).map(function(s) { return Number(s.Total || s.total) || 0; })
    };

  } catch (e) {
    Logger.log("Erro em getDashboardDataAluno_: " + e.message);
  }

  return data;
}

/**
 * Funções privadas para gerar dados de gráficos
 */

function getPerformanceChartData_(alunoId, options) {
  try {
    if (!alunoId) {
      // Performance geral de todos os alunos
      var alunos = getAllAlunos();
      return {
        labels: alunos.slice(0, 10).map(function(a) { return a.Nome || a.nome; }),
        datasets: [{
          label: 'Pontuação Média',
          data: alunos.slice(0, 10).map(function(a) {
            var ponts = getPontuacoesByAluno(a.ID || a.id);
            if (ponts.length === 0) return 0;
            return ponts.reduce(function(sum, p) { 
              return sum + (Number(p.Total) || 0); 
            }, 0) / ponts.length;
          })
        }]
      };
    } else {
      // Performance específica de um aluno
      var pontuacoes = getPontuacoesByAluno(alunoId);
      return {
        labels: pontuacoes.map(function(p, i) { return 'Simulação ' + (i + 1); }),
        datasets: [{
          label: 'Pontuação',
          data: pontuacoes.map(function(p) { return Number(p.Total || p.total) || 0; })
        }]
      };
    }
  } catch (e) {
    Logger.log("Erro em getPerformanceChartData_: " + e.message);
    return { labels: [], datasets: [] };
  }
}

function getAttendanceChartData_(options) {
  // Placeholder - implementação simplificada
  return {
    labels: ['Semana 1', 'Semana 2', 'Semana 3', 'Semana 4'],
    datasets: [{
      label: 'Frequência (%)',
      data: [95, 92, 98, 90]
    }]
  };
}

function getProgressChartData_(alunoId, options) {
  try {
    if (alunoId && typeof getPontuacoesByAluno === 'function') {
      var pontuacoes = getPontuacoesByAluno(alunoId);
      return {
        labels: pontuacoes.map(function(p, i) { return 'Avaliação ' + (i + 1); }),
        datasets: [{
          label: 'Progresso',
          data: pontuacoes.map(function(p) { return Number(p.Total || p.total) || 0; })
        }]
      };
    }
    return { labels: [], datasets: [] };
  } catch (e) {
    Logger.log("Erro em getProgressChartData_: " + e.message);
    return { labels: [], datasets: [] };
  }
}

function getComparisonChartData_(options) {
  // Placeholder - comparação entre turmas/grupos
  return {
    labels: ['Turma A', 'Turma B', 'Turma C'],
    datasets: [{
      label: 'Média Geral',
      data: [85, 78, 92]
    }]
  };
}

function getDistributionChartData_(options) {
  // Placeholder - distribuição de notas
  return {
    labels: ['0-20', '21-40', '41-60', '61-80', '81-100'],
    datasets: [{
      label: 'Número de Alunos',
      data: [2, 5, 12, 18, 8]
    }]
  };
}
