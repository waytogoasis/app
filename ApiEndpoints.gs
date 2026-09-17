// ApiEndpoints.gs
//
// Funcionalidade Principal: Expõe endpoints de API para comunicação externa, como com o Google Colab.
//
// Descrição: Este script define funções que podem ser chamadas como endpoints de API RESTful
//            ou via `google.script.run` para permitir que aplicações externas (como um notebook
//            Python no Google Colab) acessem e manipulem os dados do sistema. É crucial para
//            a integração com ferramentas de análise de dados.
//
// Integrações:
// - Google Planilha: Fonte e destino dos dados acessados via API.
// - SpreadsheetUtils.gs: Para interagir com a planilha.
// - PontuacaoService.gs, AlunoService.gs, SimulacaoService.gs: Para expor dados e funcionalidades.
// - PermissionService.gs: Para controlar o acesso aos endpoints da API.
//
// Funções Principais:
// - `handleApiPost(e)`: Handler de POST da API (delegado por Main.doPost).
// - `handleApiGet(e)`: Handler de GET da API, acionado por Main.doGet quando ?api=1.
// - `getPontuacoesForColab()`: Retorna dados de pontuação formatados para o Colab.
// - `updateAlunoStatus(alunoId, status)`: Exemplo de endpoint para atualização de dados.
//
// Observações: os pontos de entrada `doGet`/`doPost` foram CONSOLIDADOS em Main.gs.
//              Estes handlers são delegados a partir dele (evita conflito de nomes
//              reservados, em que só pode existir um doGet/doPost no projeto).

/**
 * Handler de requisições POST da API.
 * Processa requisições REST para criação e atualização de recursos.
 * @param {Object} e - Evento de requisição do Google Apps Script
 * @return {TextOutput} Resposta JSON da API
 */
function handleApiPost(e) {
  try {
    // Parse do payload
    var postData = {};
    try {
      if (e && e.postData && e.postData.contents) {
        postData = JSON.parse(e.postData.contents);
      } else if (e && e.parameter) {
        postData = e.parameter;
      }
    } catch (parseError) {
      return createApiResponse_({ success: false, error: 'Payload JSON inválido' }, 400);
    }

    var action = postData.action || e.parameter.action || '';
    var resource = postData.resource || e.parameter.resource || '';

    // Validação básica de autenticação (token ou API key)
    var authToken = postData.token || e.parameter.token || '';
    var authenticated = validateApiAuth_(authToken);
    
    if (!authenticated.success) {
      return createApiResponse_({ success: false, error: 'Autenticação inválida' }, 401);
    }

    // Roteamento baseado em resource + action
    switch (resource) {
      case 'aluno':
      case 'alunos':
        return handleAlunoPost_(postData, action, authenticated.user);
      
      case 'simulacao':
      case 'simulacoes':
        return handleSimulacaoPost_(postData, action, authenticated.user);
      
      case 'pontuacao':
      case 'pontuacoes':
        return handlePontuacaoPost_(postData, action, authenticated.user);
      
      case 'user':
      case 'usuario':
      case 'usuarios':
        return handleUsuarioPost_(postData, action, authenticated.user);
      
      default:
        return createApiResponse_({ success: false, error: 'Recurso não encontrado: ' + resource }, 404);
    }
  } catch (error) {
    Logger.log("Erro em handleApiPost: " + error.message);
    return createApiResponse_({ success: false, error: error.message }, 500);
  }
}

/**
 * Handler de requisições GET da API.
 * Processa requisições REST para leitura de recursos.
 * @param {Object} e - Evento de requisição do Google Apps Script
 * @return {TextOutput} Resposta JSON da API
 */
function handleApiGet(e) {
  try {
    var params = e.parameter || {};
    var resource = params.resource || '';
    var id = params.id || '';

    // Validação de autenticação
    var authToken = params.token || '';
    var authenticated = validateApiAuth_(authToken);
    
    if (!authenticated.success) {
      return createApiResponse_({ success: false, error: 'Autenticação inválida' }, 401);
    }

    // Roteamento baseado em resource
    switch (resource) {
      case 'aluno':
      case 'alunos':
        return handleAlunoGet_(params, authenticated.user);
      
      case 'simulacao':
      case 'simulacoes':
        return handleSimulacaoGet_(params, authenticated.user);
      
      case 'pontuacao':
      case 'pontuacoes':
        return handlePontuacaoGet_(params, authenticated.user);
      
      case 'colab':
      case 'pontuacoes_colab':
        return getPontuacoesForColab(params, authenticated.user);
      
      default:
        return createApiResponse_({ success: false, error: 'Recurso não encontrado: ' + resource }, 404);
    }
  } catch (error) {
    Logger.log("Erro em handleApiGet: " + error.message);
    return createApiResponse_({ success: false, error: error.message }, 500);
  }
}

/**
 * Retorna dados de pontuação formatados para o Google Colab.
 * Formato otimizado para análise em Python/Pandas.
 * @param {Object} params - Parâmetros da requisição
 * @param {Object} user - Usuário autenticado
 * @return {TextOutput} Resposta JSON com pontuações
 */
function getPontuacoesForColab(params, user) {
  try {
    // Verifica permissão de leitura de API
    if (typeof can === 'function' && !can(user, 'api.read')) {
      return createApiResponse_({ success: false, error: 'Permissão negada para leitura de API' }, 403);
    }

    // Parâmetros de filtro
    var alunoId = params.alunoId || params.aluno_id || '';
    var simulacaoId = params.simulacaoId || params.simulacao_id || '';
    var startDate = params.startDate || params.start_date || '';
    var endDate = params.endDate || params.end_date || '';

    var pontuacoes = [];
    
    // Busca pontuações baseado nos filtros
    if (simulacaoId && typeof getPontuacoesBySimulacao === 'function') {
      pontuacoes = getPontuacoesBySimulacao(simulacaoId);
    } else if (alunoId && typeof getPontuacoesByAluno === 'function') {
      pontuacoes = getPontuacoesByAluno(alunoId);
    } else if (typeof wtgReadObjects_ === 'function') {
      pontuacoes = wtgReadObjects_('Pontuacoes');
    } else {
      throw new Error('Serviço de pontuações não disponível');
    }

    // Enriquece dados com informações de alunos e simulações
    var enrichedData = pontuacoes.map(function(pont) {
      var record = {
        id: pont.ID || pont.id,
        simulacao_id: pont.SimulacaoID || pont.simulacaoId,
        aluno_id: pont.AlunoID || pont.alunoId,
        total: pont.Total || pont.total || 0,
        criado_em: pont.CriadoEm || pont.criadoEm || '',
        atualizado_em: pont.AtualizadoEm || pont.atualizadoEm || ''
      };

      // Parse das pontuações detalhadas
      try {
        var detalhes = typeof pont.Pontuacoes === 'string' 
          ? JSON.parse(pont.Pontuacoes) 
          : pont.Pontuacoes || {};
        record.pontuacoes = detalhes;
      } catch (e) {
        record.pontuacoes = {};
      }

      // Tenta enriquecer com dados do aluno
      try {
        if (typeof getAlunoById === 'function' && record.aluno_id) {
          var alunoResp = getAlunoById(record.aluno_id);
          if (alunoResp && alunoResp.success && alunoResp.data) {
            record.aluno_nome = alunoResp.data.Nome || alunoResp.data.nome || '';
          }
        }
      } catch (e) {}

      // Tenta enriquecer com dados da simulação
      try {
        if (typeof getSimulationById === 'function' && record.simulacao_id) {
          var simResp = getSimulationById(record.simulacao_id);
          if (simResp && simResp.success && simResp.data) {
            record.simulacao_tipo = simResp.data.Tipo || simResp.data.tipo || '';
            record.simulacao_status = simResp.data.Status || simResp.data.status || '';
          }
        }
      } catch (e) {}

      return record;
    });

    // Filtra por datas se fornecidas
    if (startDate || endDate) {
      enrichedData = enrichedData.filter(function(rec) {
        var recDate = new Date(rec.criado_em);
        if (startDate && recDate < new Date(startDate)) return false;
        if (endDate && recDate > new Date(endDate)) return false;
        return true;
      });
    }

    return createApiResponse_({
      success: true,
      data: enrichedData,
      count: enrichedData.length,
      timestamp: new Date().toISOString(),
      format: 'colab_optimized'
    }, 200);
  } catch (error) {
    Logger.log("Erro em getPontuacoesForColab: " + error.message);
    return createApiResponse_({ success: false, error: error.message }, 500);
  }
}

/**
 * Atualiza o status de um aluno via API.
 * @param {string|number} alunoId - ID do aluno
 * @param {string} status - Novo status (ativo/inativo)
 * @param {Object} user - Usuário autenticado
 * @return {TextOutput} Resposta JSON da API
 */
function updateAlunoStatus(alunoId, status, user) {
  try {
    // Verifica permissão de edição
    if (typeof can === 'function' && !can(user, 'alunos.edit')) {
      return createApiResponse_({ success: false, error: 'Permissão negada para editar alunos' }, 403);
    }

    if (!alunoId) {
      return createApiResponse_({ success: false, error: 'ID do aluno é obrigatório' }, 400);
    }

    if (!status || (status !== 'ativo' && status !== 'inativo')) {
      return createApiResponse_({ success: false, error: 'Status deve ser "ativo" ou "inativo"' }, 400);
    }

    // Atualiza o aluno
    var result;
    if (typeof updateAluno === 'function') {
      result = updateAluno(alunoId, { Status: status });
    } else if (typeof wtgUpdateRecordById_ === 'function') {
      result = wtgUpdateRecordById_('Alunos', alunoId, { Status: status });
    } else {
      throw new Error('Serviço de atualização de aluno não disponível');
    }

    // Registra auditoria
    try {
      if (typeof logAudit === 'function') {
        logAudit(
          user.id || user.ID || user.userId || 'api',
          'UPDATE_STATUS',
          'Alunos',
          alunoId,
          { oldStatus: '?', newStatus: status, via: 'API' }
        );
      }
    } catch (auditError) {
      Logger.log("Aviso: não foi possível registrar auditoria: " + auditError.message);
    }

    return createApiResponse_(result, 200);
  } catch (error) {
    Logger.log("Erro em updateAlunoStatus: " + error.message);
    return createApiResponse_({ success: false, error: error.message }, 500);
  }
}

/**
 * Funções auxiliares privadas para handlers específicos
 */

function handleAlunoPost_(data, action, user) {
  if (typeof can === 'function' && !can(user, 'alunos.' + (action === 'update' ? 'edit' : 'create'))) {
    return createApiResponse_({ success: false, error: 'Permissão negada' }, 403);
  }
  
  switch (action) {
    case 'create':
      var result = createAluno(data);
      return createApiResponse_(result, result.success ? 201 : 400);
    case 'update':
      var updateResult = updateAluno(data.id, data);
      return createApiResponse_(updateResult, updateResult.success ? 200 : 400);
    default:
      return createApiResponse_({ success: false, error: 'Ação inválida: ' + action }, 400);
  }
}

function handleAlunoGet_(params, user) {
  if (typeof can === 'function' && !can(user, 'alunos.view')) {
    return createApiResponse_({ success: false, error: 'Permissão negada' }, 403);
  }
  
  if (params.id) {
    var result = getAlunoById(params.id);
    return createApiResponse_(result, 200);
  } else {
    var all = getAllAlunos();
    return createApiResponse_({ success: true, data: all, count: all.length }, 200);
  }
}

function handleSimulacaoPost_(data, action, user) {
  if (typeof can === 'function' && !can(user, 'simulacoes.' + (action === 'end' ? 'edit' : 'create'))) {
    return createApiResponse_({ success: false, error: 'Permissão negada' }, 403);
  }
  
  switch (action) {
    case 'start':
      var result = startSimulation(data.alunoId, data.tipo, data.observacoes);
      return createApiResponse_(result, result.success ? 201 : 400);
    case 'end':
      var endResult = endSimulation(data.id, data.observacoes);
      return createApiResponse_(endResult, endResult.success ? 200 : 400);
    default:
      return createApiResponse_({ success: false, error: 'Ação inválida: ' + action }, 400);
  }
}

function handleSimulacaoGet_(params, user) {
  if (typeof can === 'function' && !can(user, 'simulacoes.view')) {
    return createApiResponse_({ success: false, error: 'Permissão negada' }, 403);
  }
  
  if (params.alunoId) {
    var result = getSimulationsByAluno(params.alunoId);
    return createApiResponse_({ success: true, data: result, count: result.length }, 200);
  } else {
    var all = getAllSimulations();
    return createApiResponse_({ success: true, data: all, count: all.length }, 200);
  }
}

function handlePontuacaoPost_(data, action, user) {
  if (typeof can === 'function' && !can(user, 'pontuacoes.create')) {
    return createApiResponse_({ success: false, error: 'Permissão negada' }, 403);
  }
  
  var result = recordPontuacao(data.simulacaoId, data.alunoId, data.pontuacoes);
  return createApiResponse_(result, result.success ? 201 : 400);
}

function handlePontuacaoGet_(params, user) {
  if (typeof can === 'function' && !can(user, 'pontuacoes.view')) {
    return createApiResponse_({ success: false, error: 'Permissão negada' }, 403);
  }
  
  if (params.simulacaoId) {
    var result = getPontuacoesBySimulacao(params.simulacaoId);
    return createApiResponse_({ success: true, data: result, count: result.length }, 200);
  } else if (params.alunoId) {
    var result = getPontuacoesByAluno(params.alunoId);
    return createApiResponse_({ success: true, data: result, count: result.length }, 200);
  } else {
    return createApiResponse_({ success: false, error: 'Parâmetro simulacaoId ou alunoId necessário' }, 400);
  }
}

function handleUsuarioPost_(data, action, user) {
  if (typeof can === 'function' && !can(user, 'users.' + (action === 'update' ? 'edit' : 'create'))) {
    return createApiResponse_({ success: false, error: 'Permissão negada' }, 403);
  }
  
  switch (action) {
    case 'create':
      var result = createUser(data);
      return createApiResponse_(result, result.success ? 201 : 400);
    case 'update':
      var updateResult = updateUser(data.id, data);
      return createApiResponse_(updateResult, updateResult.success ? 200 : 400);
    default:
      return createApiResponse_({ success: false, error: 'Ação inválida: ' + action }, 400);
  }
}

/**
 * Valida autenticação de API (token ou API key).
 */
function validateApiAuth_(token) {
  if (!token) {
    return { success: false };
  }
  
  // Método 1: Token de sessão
  try {
    if (typeof isAuthenticatedByToken === 'function' && isAuthenticatedByToken(token)) {
      var sessionUser = typeof getSessionUser === 'function' ? getSessionUser(token) : null;
      if (sessionUser) {
        return { success: true, user: sessionUser };
      }
    }
  } catch (e) {}
  
  // Método 2: API Key estática (configurada em Settings)
  try {
    if (typeof getSetting === 'function') {
      var validApiKey = getSetting('API_KEY');
      if (validApiKey && token === validApiKey) {
        return { success: true, user: { id: 'api', role: 'admin', username: 'API' } };
      }
    }
  } catch (e) {}
  
  return { success: false };
}

/**
 * Cria resposta de API padronizada com código HTTP.
 */
function createApiResponse_(data, httpCode) {
  var output = ContentService.createTextOutput(JSON.stringify(data));
  output.setMimeType(ContentService.MimeType.JSON);
  
  // Headers CORS para acesso externo
  // Nota: Apps Script Web Apps não suportam códigos HTTP customizados diretamente
  // O código HTTP é incluído no payload para interpretação do cliente
  data.httpCode = httpCode;
  
  return ContentService.createTextOutput(JSON.stringify(data))
    .setMimeType(ContentService.MimeType.JSON);
}

/**
 * @file ApiEndpoints.gs
 * Endpoints públicos chamáveis via google.script.run no frontend.
 * Gerado por codex_integrate_frontend_backend.py — pode ser customizado.
 */

/**
 * Carrega dados iniciais do dashboard.
 * Aceita o token via parâmetro (frota usa loginWithToken → ?page=app#tok= na URL).
 * @param {string} [tok] Token da sessão.
 * @return {{success:boolean, currentUser?:Object, appData?:Object, message?:string}}
 */
function getInitialAppData(tok) {
  var session = null;
  if (tok && typeof isAuthenticatedByToken === 'function' && isAuthenticatedByToken(tok)) {
    session = (typeof getSessionUser === 'function') ? getSessionUser(tok) : null;
  }
  if (!session) {
    return { success: false, message: 'Sessão inválida. Faça login novamente.' };
  }
  var currentUser = {
    id:       session.userId   || session.id       || session.username || 'unknown',
    username: session.username || session.name     || 'Usuário',
    name:     session.nome     || session.name     || session.username || 'Usuário',
    role:     session.role     || 'USER',
    email:    session.email    || ''
  };
  return { success: true, currentUser: currentUser, appData: {} };
}

/**
 * Valida se a sessão corrente ainda é válida (heartbeat do frontend).
 * @param {string} tok
 */
function pingSession(tok) {
  if (!tok) return { ok: false };
  try {
    return { ok: typeof isAuthenticatedByToken === 'function' && isAuthenticatedByToken(tok) };
  } catch (e) {
    return { ok: false };
  }
}
