/**
 * GestaoFeatures.gs — Funcionalidades autenticadas do Way To Go.
 *
 * Justificam o login: usuarios autenticados registram observacoes sobre alunos
 * e sugerem melhorias de processo. Toda acao exige credenciais validas
 * (doLogin) e e atribuida ao usuario autor.
 */

function wtg_auth_(username, password) {
  var user = doLogin(username, password); // retorna { id, role, username } ou null
  return user || null;
}

function wtg_append_(sheetName, headers, obj) {
  try {
    var ss = wtgGetSpreadsheet_();
    var sheet = ss.getSheetByName(sheetName);
    if (!sheet) {
      sheet = ss.insertSheet(sheetName);
      sheet.getRange(1, 1, 1, headers.length).setValues([headers]);
      sheet.setFrozenRows(1);
    }
    var current = sheet.getLastColumn() ? sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0].map(String) : [];
    if (!current.length) { sheet.getRange(1, 1, 1, headers.length).setValues([headers]); current = headers.slice(); }
    sheet.appendRow(current.map(function (h) { return obj[h] !== undefined ? obj[h] : ''; }));
  } catch (error) {
    Logger.log("Erro em wtg_append_: " + error.message);
    throw error; // Re-lança para tratamento superior
  }
}

function wtg_list_(sheetName) {
  var sheet = wtgGetSpreadsheet_().getSheetByName(sheetName);
  if (!sheet || sheet.getLastRow() < 2) return [];
  var values = sheet.getDataRange().getValues();
  var headers = values[0].map(String);
  return values.slice(1).map(function (r) { var o = {}; headers.forEach(function (h, i) { o[h] = r[i]; }); return o; });
}

function wtg_id_(prefix) { return prefix + '-' + Date.now() + '-' + Math.floor(Math.random() * 1000); }

/** Funcionalidade 1 — Registrar observacao sobre um aluno. */
function registrarObservacaoAluno(username, password, alunoId, observacao) {
  try {
    var user = wtg_auth_(username, password);
    if (!user) return { success: false, message: 'Credenciais invalidas.' };
    if (!String(alunoId || '').trim()) return { success: false, message: 'Informe o ID do aluno.' };
    if (!String(observacao || '').trim()) return { success: false, message: 'Informe a observacao.' };
    var id = wtg_id_('OBS');
    wtg_append_('ObservacoesAluno', ['ID', 'DataHora', 'Autor', 'AlunoID', 'Observacao'], {
      ID: id, DataHora: new Date(), Autor: user.username, AlunoID: alunoId, Observacao: observacao
    });
    return { success: true, id: id };
  } catch (error) {
    Logger.log("Erro em registrarObservacaoAluno: " + error.message);
    throw error;
  }
}

/** Funcionalidade 2 — Sugerir melhoria de processo. */
function sugerirMelhoriaProcesso(username, password, area, sugestao) {
  try {
    var user = wtg_auth_(username, password);
    if (!user) return { success: false, message: 'Credenciais invalidas.' };
    if (!String(sugestao || '').trim()) return { success: false, message: 'Informe a sugestao.' };
    var id = wtg_id_('MEL');
    wtg_append_('SugestoesMelhoria', ['ID', 'DataHora', 'Autor', 'Area', 'Sugestao', 'Status'], {
      ID: id, DataHora: new Date(), Autor: user.username, Area: area || 'geral', Sugestao: sugestao, Status: 'sugerido'
    });
    return { success: true, id: id };
  } catch (error) {
    Logger.log("Erro em sugerirMelhoriaProcesso: " + error.message);
    throw error;
  }
}

function listarObservacoesAluno(username, password) {
  if (!wtg_auth_(username, password)) return { success: false, message: 'Credenciais invalidas.' };
  return { success: true, itens: wtg_list_('ObservacoesAluno') };
}

function listarSugestoesMelhoria(username, password) {
  if (!wtg_auth_(username, password)) return { success: false, message: 'Credenciais invalidas.' };
  return { success: true, itens: wtg_list_('SugestoesMelhoria') };
}
