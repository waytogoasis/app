/**
 * MelhoriaWorkflow.gs — Workflow de aprovacao (admin) do Way To Go.
 *
 * Administradores listam sugestoes de melhoria pendentes e aprovam/rejeitam,
 * registrando revisor e data. Exige login (doLogin) com papel 'admin'. Opera
 * sobre a aba 'SugestoesMelhoria'.
 */

function wtgwf_admin_(username, password) {
  try {
    var user = doLogin(username, password); // { id, role, username } ou null
    if (!user) return null;
    return String(user.role || '').toLowerCase() === 'admin' ? user : null;
  } catch (error) {
    Logger.log("Erro em wtgwf_admin_: " + error.message);
    throw error;
  }
}

function wtgwf_update_(id, status, revisor, obs) {
  try {
    var sheet = wtgGetSpreadsheet_().getSheetByName('SugestoesMelhoria');
    if (!sheet || sheet.getLastRow() < 2) return { success: false, message: 'Nenhum registro.' };
    var headers = sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0].map(String);
    ['Revisor', 'RevisadoEm', 'ObsRevisao'].forEach(function (h) {
      if (headers.indexOf(h) === -1) { sheet.getRange(1, headers.length + 1).setValue(h); headers.push(h); }
    });
    var idCol = headers.indexOf('ID'), stCol = headers.indexOf('Status');
    var values = sheet.getRange(2, 1, sheet.getLastRow() - 1, headers.length).getValues();
    for (var i = 0; i < values.length; i++) {
      if (String(values[i][idCol]) === String(id)) {
        var row = i + 2;
        sheet.getRange(row, stCol + 1).setValue(status);
        sheet.getRange(row, headers.indexOf('Revisor') + 1).setValue(revisor);
        sheet.getRange(row, headers.indexOf('RevisadoEm') + 1).setValue(new Date());
        sheet.getRange(row, headers.indexOf('ObsRevisao') + 1).setValue(obs || '');
        return { success: true, id: id, status: status };
      }
    }
    return { success: false, message: 'ID nao encontrado: ' + id };
  } catch (error) {
    Logger.log("Erro em wtgwf_update_: " + error.message);
    throw error; // Re-lança para tratamento superior
  }
}

function wtgwf_list_(statusFiltro) {
  try {
    try {
      var sheet = wtgGetSpreadsheet_().getSheetByName('SugestoesMelhoria');
      if (!sheet || sheet.getLastRow() < 2) return [];
      var values = sheet.getDataRange().getValues();
      var headers = values[0].map(String);
      return values.slice(1).map(function (r) { var o = {}; headers.forEach(function (h, i) { o[h] = r[i]; }); return o; })
        .filter(function (o) { return !statusFiltro || String(o.Status || '').toLowerCase() === statusFiltro; });
    } catch (error) {
      Logger.log("Erro em wtgwf_list_: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em wtgwf_list_: " + error.message);
    throw error;
  }
}

function listarMelhoriasPendentes(username, password) {
  if (!wtgwf_admin_(username, password)) return { success: false, message: 'Acesso restrito a administradores.' };
  return { success: true, itens: wtgwf_list_('sugerido') };
}

function aprovarMelhoria(username, password, id, observacao) {
  try {
    var admin = wtgwf_admin_(username, password);
    if (!admin) return { success: false, message: 'Acesso restrito a administradores.' };
    if (!String(id || '').trim()) return { success: false, message: 'Informe o ID.' };
    return wtgwf_update_(id, 'aprovado', admin.username, observacao);
  } catch (error) {
    Logger.log("Erro em aprovarMelhoria: " + error.message);
    throw error;
  }
}

function rejeitarMelhoria(username, password, id, motivo) {
  try {
    var admin = wtgwf_admin_(username, password);
    if (!admin) return { success: false, message: 'Acesso restrito a administradores.' };
    if (!String(id || '').trim()) return { success: false, message: 'Informe o ID.' };
    if (!String(motivo || '').trim()) return { success: false, message: 'Informe o motivo da rejeicao.' };
    return wtgwf_update_(id, 'rejeitado', admin.username, motivo);
  } catch (error) {
    Logger.log("Erro em rejeitarMelhoria: " + error.message);
    throw error;
  }
}
