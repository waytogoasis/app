/**
 * ApiGateway.gs — Gateway RPC unico da frota (consolidado em 2026-06-21).
 *
 * O ApiClient.html chama `google.script.run.apiCall(service, method, payload)`.
 * Este e o UNICO ponto de entrada `apiCall` do projeto (a colisao com
 * ApiGatewayStandard.gs foi eliminada).
 *
 * Login e sessao sao resolvidos por DESCOBERTA EM RUNTIME (typeof), espelhando
 * o FleetLoginCheck.gs (harness verde) — assim o gateway funciona com a funcao
 * de auth real de cada projeto sem hardcode. Texto plano no login (decisao de
 * frota): apenas delega; nao verifica nem gera hash.
 *
 * Convencao: switch(`${service}.${method}`); envelope de sucesso { ok:true, data },
 * de falha { ok:false, error:{ message } }. Unica rota publica: AuthService.login.
 */
function apiCall(service, method, payload) {
  try {
    var request = payload || {};
    var operation = String(service) + '.' + String(method);
    var publicCall = operation === 'AuthService.login';

    try {
      var principal = publicCall ? null : gw_principalFromRequest_(request);
      if (!publicCall && !principal) {
        throw new Error('Sessao expirada. Faca login novamente.');
      }

      var data;
      switch (operation) {
        case 'AuthService.login':
          data = gw_login_(
            String(request.username || '').trim(),
            String(request.password || '')
          );
          if (!gw_isLoginOk_(data)) throw new Error('Usuario ou senha invalidos.');
          data = gw_normalizeLogin_(data);
          break;
        case 'AuthService.logout':
          data = gw_logout_(request.authToken || request.token);
          break;
        case 'Students.list':
          gw_requireRole_(principal, ['admin', 'professor']);
          data = gw_studentsForClient_();
          break;
        case 'Simulations.list':
          gw_requireRole_(principal, ['admin', 'professor']);
          data = gw_simulationsForClient_();
          break;
        case 'Simulations.create':
          gw_requireRole_(principal, ['admin', 'professor']);
          data = gw_createSimulation_(request, principal);
          break;
        case 'Simulations.finish':
          gw_requireRole_(principal, ['admin', 'professor']);
          data = gw_finishSimulation_(request);
          break;
        case 'Citizenship.scenario':
          data = getCitizenshipDecisionScenario();
          break;
        case 'Citizenship.submit':
          data = submitCitizenshipDecision(principal.id, request);
          break;
        case 'Citizenship.reflect':
          data = completeCitizenshipReflection(principal.id, request);
          break;
        case 'Citizenship.progress':
          data = getCitizenshipDecisionProgress(principal.id);
          break;
        case 'StudentDashboard.get':
          data = getAlunoDashboardData(principal.id);
          break;
        default:
          throw new Error('Operacao de API nao permitida: ' + operation);
      }

      return { ok: true, data: toClientSafe_(data) };
    } catch (apiError) {
      return {
        ok: false,
        error: { message: (apiError && apiError.message) || 'Erro interno do servidor.' }
      };
    }
  } catch (error) {
    Logger.log("Erro em apiCall: " + error.message);
    throw error;
  }
}

/**
 * Resolve o login na MESMA ordem do FleetLoginCheck.gs: a primeira funcao de
 * login existente vence. Tenta a forma posicional (u, p) e, se falhar, a forma
 * de objeto ({ username, password, senha, email }). Cobre as duas convencoes da frota.
 */
function gw_login_(username, password) {
  try {
    var entries = [];
    if (typeof AuthService !== 'undefined' && AuthService && typeof AuthService.login === 'function') {
      entries.push(function (form) { return AuthService.login.apply(AuthService, form); });
    }
    if (typeof loginWithPassword === 'function')   entries.push(function (form) { return loginWithPassword.apply(null, form); });
    if (typeof loginWithToken === 'function')      entries.push(function (form) { return loginWithToken.apply(null, form); });
    if (typeof processLoginRequest === 'function') entries.push(function (form) { return processLoginRequest.apply(null, form); });
    if (typeof doLogin === 'function')             entries.push(function (form) { return doLogin.apply(null, form); });
    if (typeof login === 'function')               entries.push(function (form) { return login.apply(null, form); });
    if (typeof authenticate === 'function')        entries.push(function (form) { return authenticate.apply(null, form); });

    var positional = [username, password];
    var objectForm = [{ username: username, password: password, senha: password, email: username }];
    var last = null;
    for (var i = 0; i < entries.length; i++) {
      try {
        var r = entries[i](positional);
        if (gw_isLoginOk_(r)) return r;
        last = r;
        try {
          var r2 = entries[i](objectForm);
          if (gw_isLoginOk_(r2)) return r2;
          last = last || r2;
        } catch (ignoredObj) {}
      } catch (err) {
        last = last || { success: false, message: (err && err.message) || String(err) };
      }
    }
    return last;
  } catch (error) {
    Logger.log("Erro em gw_login_: " + error.message);
    throw error;
  }
}

/** Resolve o logout e propaga falhas de revogação ao envelope do gateway. */
function gw_logout_(token) {
  try {
    if (token && typeof logoutWithToken === 'function') return logoutWithToken(token);
    if (typeof doLogout === 'function') return doLogout();
    if (typeof logout === 'function') return logout();
  } catch (error) {
    throw new Error('Não foi possível confirmar o encerramento da sessão. Tente novamente.');
  }
  throw new Error('Serviço de encerramento de sessão indisponível.');
}

/** Resolve exclusivamente a sessao apresentada pela requisicao do browser. */
function gw_principalFromRequest_(request) {
  var token = String((request && (request.authToken || request.token)) || '').trim();
  if (!token || typeof isAuthenticatedByToken !== 'function' || !isAuthenticatedByToken(token)) {
    return null;
  }
  if (typeof getSessionUser !== 'function') return null;
  var user = getSessionUser(token);
  if (!user) return null;
  return {
    id: String(user.id || user.userId || user.username || ''),
    username: String(user.username || user.id || ''),
    role: String(user.role || user.perfil || 'aluno').toLowerCase()
  };
}

function gw_requireRole_(principal, allowedRoles) {
  if (!principal || allowedRoles.indexOf(String(principal.role || '').toLowerCase()) === -1) {
    throw new Error('Permissao insuficiente para esta operacao.');
  }
}

function gw_studentsForClient_() {
  if (typeof getAllAlunos !== 'function') throw new Error('Servico de alunos indisponivel.');
  return (getAllAlunos() || []).map(function (student) {
    return {
      id: String(student.ID || student.id || ''),
      name: String(student.Nome || student.nome || student.nomeAluno || '')
    };
  }).filter(function (student) { return student.id && student.name; });
}

function gw_simulationsForClient_() {
  if (typeof getAllSimulations !== 'function') throw new Error('Servico de simulacoes indisponivel.');
  return (getAllSimulations() || []).map(gw_simulationDto_);
}

function gw_createSimulation_(request, principal) {
  var studentIds = Array.isArray(request.studentIds) ? request.studentIds : [];
  studentIds = studentIds.map(function (id) { return String(id || '').trim(); })
    .filter(function (id, index, all) { return id && all.indexOf(id) === index; });
  var simulationType = String(request.simulationType || '').trim();
  if (!studentIds.length) throw new Error('Selecione ao menos um aluno.');
  if (!simulationType) throw new Error('Informe o tipo de simulacao.');

  var known = {};
  gw_studentsForClient_().forEach(function (student) { known[student.id] = true; });
  if (studentIds.some(function (id) { return !known[id]; })) {
    throw new Error('A selecao contem aluno inexistente ou inativo.');
  }

  var notes = String(request.notes || '').trim();
  var auditNote = 'Criada por ' + String(principal.username || principal.id);
  var result = startSimulation(
    JSON.stringify(studentIds),
    simulationType,
    notes ? notes + ' | ' + auditNote : auditNote,
    String(request.scheduledDate || '').trim()
  );
  if (!result || result.success === false) {
    throw new Error((result && result.message) || 'Nao foi possivel iniciar a simulacao.');
  }
  var dto = gw_simulationDto_(result.data || result);
  return dto;
}

function gw_finishSimulation_(request) {
  var simulationId = String(request.simulationId || '').trim();
  if (!simulationId) throw new Error('simulationId obrigatorio.');
  var result = endSimulation(simulationId, String(request.notes || '').trim());
  if (!result || result.success === false) {
    throw new Error((result && result.message) || 'Nao foi possivel finalizar a simulacao.');
  }
  return gw_simulationDto_(result.data || result);
}

function gw_simulationDto_(simulation) {
  simulation = simulation || {};
  var rawStudentIds = simulation.AlunoID || simulation.alunoId || simulation.alunoIds || [];
  var studentIds = [];
  if (Array.isArray(rawStudentIds)) {
    studentIds = rawStudentIds;
  } else {
    try { studentIds = JSON.parse(String(rawStudentIds || '[]')); }
    catch (ignoredJson) { studentIds = String(rawStudentIds || '').split(','); }
  }
  return {
    id: String(simulation.ID || simulation.id || ''),
    studentIds: studentIds.map(function (id) { return String(id).trim(); }).filter(Boolean),
    simulationType: String(simulation.Tipo || simulation.tipo || simulation.tipoSimulacao || ''),
    status: String(simulation.Status || simulation.status || ''),
    notes: String(simulation.Observacoes || simulation.observacoes || ''),
    startedAt: simulation.IniciadoEm || simulation.iniciadoEm || simulation.dataSimulacao || '',
    finishedAt: simulation.FinalizadoEm || simulation.finalizadoEm || ''
  };
}

/**
 * Verificador de sessao da frota (fail-closed): primeira funcao existente vence.
 * Cobre as variantes nativas (sgteLegacy, AuthService, getCurrentUser*).
 */
function gw_currentUser_() {
  try {
    if (typeof getCurrentSessionUser === 'function')         { var a = getCurrentSessionUser();          if (a) return a; }
    if (typeof getCurrentSessionUser_sgteLegacy === 'function') { var b = getCurrentSessionUser_sgteLegacy(); if (b) return b; }
    if (typeof AuthService !== 'undefined' && AuthService && typeof AuthService.getSessionUser === 'function') { var c = AuthService.getSessionUser(); if (c) return c; }
    if (typeof getCurrentUser_ === 'function')               { var d = getCurrentUser_();                if (d) return d; }
    if (typeof getCurrentUser === 'function')                { var e = getCurrentUser();                 if (e) return e; }
  } catch (ignored) {}
  return null;
}

/** Normaliza o resultado do login para um booleano de sucesso (igual ao harness). */
function gw_isLoginOk_(r) {
  if (r === null || r === undefined || r === false) return false;
  if (typeof r === 'object') {
    if (r.success === false || r.ok === false) return false;
    if (r.success === true || r.ok === true) return true;
    if (r.token || r.sessionToken || r.redirectUrl || r.session) return true;
    if (r.user || r.id || r.username || r.role || r.perfil) return true;
    return false;
  }
  return !!r;
}


/**
 * Normaliza qualquer resultado de login aceito por gw_isLoginOk_ para o
 * envelope { success:true, user?:{}, token?:string } esperado pelo Login.html.
 * Sem esta etapa, funcoes que retornam { ok:true } ou o objeto de usuario
 * diretamente passam a validacao mas chegam ao cliente sem .success=true.
 */
function gw_normalizeLogin_(r) {
  if (!r || typeof r !== 'object') return { success: true };
  if ('success' in r) return r;
  if ('ok' in r) {
    var out = { success: !!r.ok };
    if (r.user)         out.user         = r.user;
    if (r.principal)    out.user         = r.principal;
    if (r.token)        out.token        = r.token;
    if (r.sessionToken) out.sessionToken = r.sessionToken;
    if (r.message)      out.message      = r.message;
    return out;
  }
  if (r.id || r.username || r.role || r.perfil) return { success: true, user: r };
  if (r.token || r.sessionToken) return { success: true, token: r.token || r.sessionToken };
  return { success: true };
}

/** Sanitiza dados para o cliente: remove credenciais, serializa Date, recursivo. */
function toClientSafe_(value) {
  try {
    if (value instanceof Date) return value.toISOString();
    if (Array.isArray(value)) return value.map(toClientSafe_);
    if (value && typeof value === 'object') {
      var safe = {};
      Object.keys(value).forEach(function (key) {
        if (key === 'password' || key === 'passwordHash' || key === 'senha' || key === 'senha_hash') return;
        safe[key] = toClientSafe_(value[key]);
      });
      return safe;
    }
    return value;
  } catch (error) {
    Logger.log("Erro em toClientSafe_: " + error.message);
    throw error;
  }
}
