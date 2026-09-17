// ScoreManagementUI.gs
//
// Funcionalidade Principal: Funções de Apps Script para interagir com a interface de gerenciamento de pontuações.
//
// Descrição: Este script atua como uma ponte entre o frontend HTML de gerenciamento de pontuações
//            e o backend `PontuacaoService.gs`. Ele recebe requisições da UI, chama as funções
//            apropriadas do `PontuacaoService.gs` e retorna os resultados para a interface.
//
// Integrações:
// - PontuacaoService.gs: Para realizar operações CRUD de pontuações.
// - HtmlService.gs: Para servir a página `PontuacaoForm.html`.
// - PermissionService.gs: Para verificar permissões antes de executar ações.
//
// Funções Principais:
// - `getScoresForUI(simulationId)`: Retorna uma lista de pontuações para exibição na UI.
// - `saveScoreFromUI(scoreData)`: Salva (cria ou atualiza) uma pontuação a partir dos dados da UI.
// - `deleteScoreFromUI(scoreId)`: Deleta uma pontuação a partir da UI.
//
// Observações: Garante que as interações da interface do usuário com o backend sejam seguras e eficientes.

/**
 * Retorna lista de pontuações para exibição na UI.
 * @param {string|number} [simulationId] - ID da simulação (opcional, retorna todas se não fornecido)
 * @param {Object} [filters] - Filtros adicionais (alunoId, dataInicio, dataFim)
 * @return {Object} Lista de pontuações com metadata
 */
function getScoresForUI(simulationId, filters) {
  try {
    // Verifica permissão
    var currentUser = scoreManagementCurrentUser_();
    if (typeof can === 'function' && !can(currentUser, 'pontuacoes.view')) {
      return {
        success: false,
        error: 'Permissão negada para visualizar pontuações'
      };
    }

    filters = filters || {};
    var pontuacoes = [];

    // Obtém pontuações baseado nos parâmetros
    try {
      if (simulationId && typeof getPontuacoesBySimulacao === 'function') {
        pontuacoes = getPontuacoesBySimulacao(simulationId);
      } else if (filters.alunoId && typeof getPontuacoesByAluno === 'function') {
        pontuacoes = getPontuacoesByAluno(filters.alunoId);
      } else if (typeof wtgReadObjects_ === 'function') {
        pontuacoes = wtgReadObjects_('Pontuacoes');
      } else {
        throw new Error('Serviço de pontuações não disponível');
      }
    } catch (e) {
      Logger.log("Erro ao obter pontuações: " + e.message);
      return { success: false, error: 'Erro ao buscar pontuações: ' + e.message };
    }

    // Aplica filtros adicionais
    if (filters.dataInicio) {
      var dataInicio = new Date(filters.dataInicio);
      pontuacoes = pontuacoes.filter(function(p) {
        var dataPont = new Date(p.CriadoEm || p.criadoEm || 0);
        return dataPont >= dataInicio;
      });
    }

    if (filters.dataFim) {
      var dataFim = new Date(filters.dataFim);
      pontuacoes = pontuacoes.filter(function(p) {
        var dataPont = new Date(p.CriadoEm || p.criadoEm || 0);
        return dataPont <= dataFim;
      });
    }

    // Enriquece dados com informações de alunos e simulações
    var enrichedPontuacoes = pontuacoes.map(function(pont) {
      var enriched = {
        id: pont.ID || pont.id,
        simulacaoId: pont.SimulacaoID || pont.simulacaoId,
        alunoId: pont.AlunoID || pont.alunoId,
        total: pont.Total || pont.total || 0,
        criadoEm: pont.CriadoEm || pont.criadoEm || '',
        atualizadoEm: pont.AtualizadoEm || pont.atualizadoEm || ''
      };

      // Parse das pontuações detalhadas
      try {
        var detalhes = typeof pont.Pontuacoes === 'string' 
          ? JSON.parse(pont.Pontuacoes) 
          : pont.Pontuacoes || {};
        enriched.pontuacoes = detalhes;
      } catch (e) {
        enriched.pontuacoes = {};
      }

      // Enriquece com dados do aluno
      try {
        if (typeof getAlunoById === 'function') {
          var alunoResp = getAlunoById(enriched.alunoId);
          if (alunoResp && alunoResp.success && alunoResp.data) {
            enriched.alunoNome = alunoResp.data.Nome || alunoResp.data.nome || '';
          }
        }
      } catch (e) {
        enriched.alunoNome = 'Desconhecido';
      }

      // Enriquece com dados da simulação
      try {
        if (typeof getSimulationById === 'function') {
          var simResp = getSimulationById(enriched.simulacaoId);
          if (simResp && simResp.success && simResp.data) {
            enriched.simulacaoTipo = simResp.data.Tipo || simResp.data.tipo || '';
            enriched.simulacaoStatus = simResp.data.Status || simResp.data.status || '';
          }
        }
      } catch (e) {
        enriched.simulacaoTipo = 'Desconhecido';
      }

      return enriched;
    });

    // Ordena por data de criação (mais recentes primeiro)
    enrichedPontuacoes.sort(function(a, b) {
      var dateA = new Date(a.criadoEm || 0);
      var dateB = new Date(b.criadoEm || 0);
      return dateB - dateA;
    });

    return {
      success: true,
      data: enrichedPontuacoes,
      count: enrichedPontuacoes.length,
      timestamp: new Date().toISOString()
    };
  } catch (error) {
    Logger.log("Erro em getScoresForUI: " + error.message);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Salva (cria ou atualiza) uma pontuação a partir dos dados da UI.
 * @param {Object} scoreData - Dados da pontuação (com ou sem ID)
 * @return {Object} Resultado da operação
 */
function saveScoreFromUI(scoreData) {
  try {
    // Verifica permissão
    var currentUser = scoreManagementCurrentUser_();
    var isUpdate = !!(scoreData.id || scoreData.ID);
    var action = isUpdate ? 'pontuacoes.edit' : 'pontuacoes.create';
    
    if (typeof can === 'function' && !can(currentUser, action)) {
      return {
        success: false,
        error: 'Permissão negada para ' + (isUpdate ? 'editar' : 'criar') + ' pontuações'
      };
    }

    if (!scoreData) {
      return { success: false, error: 'Dados da pontuação não fornecidos' };
    }

    // Validação básica
    if (!scoreData.simulacaoId && !scoreData.SimulacaoID) {
      return { success: false, error: 'ID da simulação é obrigatório' };
    }

    if (!scoreData.alunoId && !scoreData.AlunoID) {
      return { success: false, error: 'ID do aluno é obrigatório' };
    }

    var result;

    if (isUpdate) {
      // Atualização de pontuação existente
      var pontuacaoId = scoreData.id || scoreData.ID;
      
      // Prepara objeto de pontuações
      var pontuacoes = scoreData.pontuacoes || {};
      if (typeof pontuacoes === 'string') {
        try {
          pontuacoes = JSON.parse(pontuacoes);
        } catch (e) {
          pontuacoes = {};
        }
      }

      var updateData = {
        Pontuacoes: JSON.stringify(pontuacoes)
      };

      // Recalcula total se função estiver disponível
      if (typeof pontuacaoTotal_ === 'function') {
        updateData.Total = pontuacaoTotal_(pontuacoes);
      } else {
        // Cálculo simplificado de média
        var values = Object.keys(pontuacoes).map(function(k) { 
          return Number(pontuacoes[k]) || 0; 
        });
        if (values.length > 0) {
          var sum = values.reduce(function(a, b) { return a + b; }, 0);
          updateData.Total = Math.round(sum / values.length * 100) / 100;
        }
      }

      if (typeof updatePontuacao === 'function') {
        result = updatePontuacao(pontuacaoId, pontuacoes);
      } else if (typeof wtgUpdateRecordById_ === 'function') {
        result = wtgUpdateRecordById_('Pontuacoes', pontuacaoId, updateData);
      } else {
        throw new Error('Serviço de atualização de pontuação não disponível');
      }

      // Registra auditoria
      try {
        if (typeof logAudit === 'function') {
          logAudit(
            currentUser.id || currentUser.ID || 'user',
            'UPDATE',
            'Pontuacoes',
            pontuacaoId,
            { updated: updateData }
          );
        }
      } catch (auditError) {
        Logger.log("Aviso: não foi possível registrar auditoria: " + auditError.message);
      }

    } else {
      // Criação de nova pontuação
      var simulacaoId = scoreData.simulacaoId || scoreData.SimulacaoID;
      var alunoId = scoreData.alunoId || scoreData.AlunoID;
      var pontuacoesData = scoreData.pontuacoes || scoreData.Pontuacoes || {};
      
      if (typeof pontuacoesData === 'string') {
        try {
          pontuacoesData = JSON.parse(pontuacoesData);
        } catch (e) {
          pontuacoesData = {};
        }
      }

      if (typeof recordPontuacao === 'function') {
        result = recordPontuacao(simulacaoId, alunoId, pontuacoesData);
      } else {
        throw new Error('Serviço de criação de pontuação não disponível');
      }

      // Registra auditoria
      try {
        if (typeof logAudit === 'function' && result.success) {
          logAudit(
            currentUser.id || currentUser.ID || 'user',
            'CREATE',
            'Pontuacoes',
            result.data ? (result.data.ID || result.data.id) : 'new',
            { 
              simulacaoId: simulacaoId,
              alunoId: alunoId,
              total: result.data ? result.data.Total : 0
            }
          );
        }
      } catch (auditError) {
        Logger.log("Aviso: não foi possível registrar auditoria: " + auditError.message);
      }
    }

    return result;
  } catch (error) {
    Logger.log("Erro em saveScoreFromUI: " + error.message);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Deleta uma pontuação a partir da UI.
 * @param {string|number} scoreId - ID da pontuação a ser deletada
 * @return {Object} Resultado da operação
 */
function deleteScoreFromUI(scoreId) {
  try {
    // Verifica permissão
    var currentUser = scoreManagementCurrentUser_();
    if (typeof can === 'function' && !can(currentUser, 'pontuacoes.delete')) {
      return {
        success: false,
        error: 'Permissão negada para deletar pontuações'
      };
    }

    if (!scoreId) {
      return { success: false, error: 'ID da pontuação não fornecido' };
    }

    // Verifica se a pontuação existe
    var pontuacao = null;
    try {
      if (typeof wtgFindRecordById_ === 'function') {
        var pontResult = wtgFindRecordById_('Pontuacoes', scoreId);
        if (pontResult && pontResult.success) {
          pontuacao = pontResult.data;
        }
      }
    } catch (e) {
      Logger.log("Aviso: não foi possível verificar existência da pontuação: " + e.message);
    }

    if (!pontuacao) {
      return { success: false, error: 'Pontuação não encontrada: ' + scoreId };
    }

    // Hard delete (remoção permanente) - pontuações podem ser recriadas
    var result;
    if (typeof wtgDeleteRecord_ === 'function') {
      result = wtgDeleteRecord_('Pontuacoes', scoreId);
    } else if (typeof wtgUpdateRecordById_ === 'function') {
      // Fallback: marca como deletada
      result = wtgUpdateRecordById_('Pontuacoes', scoreId, { 
        Deleted: true,
        DeletedAt: new Date().toISOString()
      });
    } else {
      throw new Error('Serviço de deleção de pontuação não disponível');
    }

    // Registra auditoria
    try {
      if (typeof logAudit === 'function' && result.success) {
        logAudit(
          currentUser.id || currentUser.ID || 'user',
          'DELETE',
          'Pontuacoes',
          scoreId,
          { 
            simulacaoId: pontuacao.SimulacaoID || pontuacao.simulacaoId,
            alunoId: pontuacao.AlunoID || pontuacao.alunoId,
            total: pontuacao.Total || pontuacao.total
          }
        );
      }
    } catch (auditError) {
      Logger.log("Aviso: não foi possível registrar auditoria: " + auditError.message);
    }

    return result;
  } catch (error) {
    Logger.log("Erro em deleteScoreFromUI: " + error.message);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Função auxiliar para obter usuário atual (reutilizada de StudentManagementUI)
 */
function scoreManagementCurrentUser_() {
  try {
    if (typeof getCurrentSessionUser === 'function') {
      return getCurrentSessionUser();
    } else if (typeof getSession === 'function') {
      var session = getSession();
      if (session && session.userId && typeof getUserById === 'function') {
        return getUserById(session.userId);
      }
    }
    return { id: 'unknown', role: 'user' };
  } catch (e) {
    return { id: 'unknown', role: 'user' };
  }
}
