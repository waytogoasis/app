// UserService.gs
//
// Funcionalidade Principal: Gerencia as operações CRUD (Criar, Ler, Atualizar, Deletar) para usuários no sistema.
//
// Descrição: Este script fornece funções para interagir com a aba 'Usuarios' da Google Planilha,
//            permitindo o cadastro de novos usuários, a recuperação de informações de usuários
//            existentes, a atualização de seus dados e a remoção de registros.
//            É essencial para a administração de quem pode acessar o sistema.
//
// Integrações:
// - Google Planilha (aba 'Usuarios'): Todas as operações de dados são realizadas nesta aba.
// - SpreadsheetUtils.gs: Utiliza funções auxiliares para manipulação da planilha.
// - AuthService.gs: Pode ser chamado para verificar permissões antes de executar operações sensíveis.
//
// Funções Principais:
// - `createUser(userData)`: Adiciona um novo usuário à planilha.
// - `getUserById(userId)`: Retorna os dados de um usuário específico.
// - `updateUser(userId, newUserData)`: Atualiza as informações de um usuário.
// - `deleteUser(userId)`: Remove um usuário da planilha.
// - `getAllUsers()`: Retorna uma lista de todos os usuários cadastrados.
//
// Observações: As senhas são armazenadas em texto plano, conforme requisito. A validação de dados
//              de entrada deve ser robusta para evitar inconsistências na planilha.

function createUser(userData) {
  // Aba unica 'Usuarios' com cabecalho canonico (migra colunas de 'Users' e do
  // antigo placeholder). Usa 'Password' (texto plano) — o mesmo campo que o login
  // (getAllUsers/verifyPassword_) e o seeding sintetico leem; antes era 'Senha',
  // que o registro (envia 'password') nunca preenchia.
  return wtgCreateRecord_('Usuarios', ['ID', 'Username', 'Password', 'Role', 'Nome', 'Email', 'Status', 'LastLoginAt', 'CriadoEm', 'AtualizadoEm'], userData, {
    required: ['Username', 'Password'],
    defaults: { Role: 'professor', Status: 'ativo' }
  });
}

function getUserById(userId) {
  return wtgFindRecordById_('Usuarios', userId);
}

function updateUser(userId, newUserData) {
  return wtgUpdateRecordById_('Usuarios', userId, newUserData);
}

function deleteUser(userId) {
  return wtgUpdateRecordById_('Usuarios', userId, { Status: 'inativo', Ativo: false });
}

function getAllUsers() {
  try {
    try {
      var ss = wtgGetSpreadsheet_();
      if (!ss) return [];
      var sheet = ss.getSheetByName('Usuarios');
      if (!sheet || sheet.getLastRow() < 2) return [];

      var values = sheet.getDataRange().getValues();
      var headers = values[0].map(function (h) { return String(h || '').trim(); });
      return values.slice(1)
        .filter(function (row) { return row.some(function (c) { return c !== '' && c !== null; }); })
        .map(function (row) {
          var obj = {};
          headers.forEach(function (h, i) {
            obj[h] = row[i];
            obj[String(h).toLowerCase()] = row[i]; // chaves minusculas usadas por doLogin/verifyPassword_
          });
          return obj;
        });
    } catch (error) {
      Logger.log("Erro em getAllUsers: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em getAllUsers: " + error.message);
    throw error;
  }
}

/** Resolve a planilha principal (CONFIG/Config/Script Property, fallback ativa). */
function wtgGetSpreadsheet_() {
  try {
    try {
      if (typeof CONFIG !== 'undefined' && CONFIG.SPREADSHEET_ID) return SpreadsheetApp.openById(CONFIG.SPREADSHEET_ID);
      if (typeof Config !== 'undefined' && Config.SPREADSHEET_ID) return SpreadsheetApp.openById(Config.SPREADSHEET_ID);
      var props = PropertiesService.getScriptProperties();
      var id = props.getProperty('SPREADSHEET_ID') || props.getProperty('SPREADSHEETS_ID');
      if (id) return SpreadsheetApp.openById(id);
      return getBoundSpreadsheet_();
    } catch (error) {
      Logger.log("Erro em wtgGetSpreadsheet_: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em wtgGetSpreadsheet_: " + error.message);
    throw error;
  }
}

function wtgWithWriteLock_(operationName, callback) {
  try {
    var lock = LockService.getScriptLock();
    if (!lock.tryLock(10000)) throw new Error('Nao foi possivel obter lock para ' + operationName + '.');
    try {
      return callback();
    } finally {
      lock.releaseLock();
    }
  } catch (error) {
    Logger.log("Erro em wtgWithWriteLock_: " + error.message);
    throw error;
  }
}

function wtgEnsureSheet_(sheetName, headers) {
  try {
    var ss = wtgGetSpreadsheet_();
    var sheet = ss.getSheetByName(sheetName);
    if (!sheet) sheet = ss.insertSheet(sheetName);
    if (sheet.getLastRow() === 0) {
      sheet.getRange(1, 1, 1, headers.length).setValues([headers]);
      sheet.setFrozenRows(1);
    }
    return sheet;
  } catch (error) {
    Logger.log("Erro em wtgEnsureSheet_: " + error.message);
    throw error;
  }
}

function wtgReadObjects_(sheetName) {
  try {
    var ss = wtgGetSpreadsheet_();
    var sheet = ss && ss.getSheetByName(sheetName);
    if (!sheet || sheet.getLastRow() < 2) return [];
    var values = sheet.getDataRange().getValues();
    var headers = values[0].map(function (h) { return String(h || '').trim(); });
    return values.slice(1).filter(function (row) {
      return row.some(function (cell) { return cell !== '' && cell !== null; });
    }).map(function (row) {
      var obj = {};
      headers.forEach(function (h, i) {
        obj[h] = row[i];
        obj[String(h).toLowerCase()] = row[i];
      });
      return obj;
    });
  } catch (error) {
    Logger.log("Erro em wtgReadObjects_: " + error.message);
    throw error;
  }
}

function wtgNormalizeRecord_(headers, data, options) {
  try {
    options = options || {};
    data = data || {};
    var defaults = options.defaults || {};
    var record = {};
    headers.forEach(function (h) {
      var lower = String(h).toLowerCase();
      var value = data[h];
      if (value === undefined) value = data[lower];
      if (value === undefined) value = defaults[h];
      if (value === undefined) value = defaults[lower];
      record[h] = value !== undefined ? value : '';
    });
    if (!record.ID && !record.id) record[headers[0]] = Utilities.getUuid();
    if (headers.indexOf('CriadoEm') >= 0 && !record.CriadoEm) record.CriadoEm = new Date().toISOString();
    if (headers.indexOf('AtualizadoEm') >= 0) record.AtualizadoEm = new Date().toISOString();
    return record;
  } catch (error) {
    Logger.log("Erro em wtgNormalizeRecord_: " + error.message);
    throw error;
  }
}

function wtgAssertRequired_(data, required) {
  try {
    (required || []).forEach(function (field) {
      var value = data[field] !== undefined ? data[field] : data[String(field).toLowerCase()];
      if (String(value || '').trim() === '') throw new Error('Campo obrigatorio ausente: ' + field);
    });
  } catch (error) {
    Logger.log("Erro em wtgAssertRequired_: " + error.message);
    throw error;
  }
}

function wtgCreateRecord_(sheetName, headers, data, options) {
  try {
    try {
      options = options || {};
      wtgAssertRequired_(data || {}, options.required);
      var sheet = wtgEnsureSheet_(sheetName, headers);
      var currentHeaders = sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0].map(String);
      var record = wtgNormalizeRecord_(currentHeaders, data, options);
      wtgWithWriteLock_('create:' + sheetName, function () {
        sheet.appendRow(currentHeaders.map(function (h) { return record[h]; }));
      });
      return { success: true, data: record };
    } catch (error) {
      Logger.log("Erro em wtgCreateRecord_: " + error.message);
      throw error; // Re-lança para tratamento superior
    }
  } catch (error) {
    Logger.log("Erro em wtgCreateRecord_: " + error.message);
    throw error;
  }
}

function wtgFindRecordById_(sheetName, id) {
  var items = wtgReadObjects_(sheetName);
  var found = items.filter(function (item) {
    return String(item.ID || item.id || '') === String(id || '');
  })[0] || null;
  return { success: !!found, data: found, message: found ? '' : 'Registro nao encontrado.' };
}

function wtgUpdateRecordById_(sheetName, id, updates) {
  try {
    try {
      try {
        var sheet = wtgGetSpreadsheet_().getSheetByName(sheetName);
        if (!sheet || sheet.getLastRow() < 2) return { success: false, message: 'Registro nao encontrado.' };
        var values = sheet.getDataRange().getValues();
        var headers = values[0].map(String);
        var idCol = headers.map(function (h) { return h.toLowerCase(); }).indexOf('id');
        if (idCol < 0) idCol = 0;
        for (var i = 1; i < values.length; i++) {
          if (String(values[i][idCol]) === String(id)) {
            var row = values[i].slice();
            headers.forEach(function (h, col) {
              var lower = h.toLowerCase();
              if (updates[h] !== undefined) row[col] = updates[h];
              else if (updates[lower] !== undefined) row[col] = updates[lower];
            });
            var updatedAt = headers.indexOf('AtualizadoEm');
            if (updatedAt >= 0) row[updatedAt] = new Date().toISOString();
            wtgWithWriteLock_('update:' + sheetName, function () {
              sheet.getRange(i + 1, 1, 1, headers.length).setValues([row.slice(0, headers.length)]);
            });
            var record = {};
            headers.forEach(function (h, col) { record[h] = row[col]; record[h.toLowerCase()] = row[col]; });
            return { success: true, data: record };
          }
        }
        return { success: false, message: 'Registro nao encontrado.' };
      } catch (error) {
        Logger.log("Erro em wtgUpdateRecordById_: " + error.message);
        throw error; // Re-lança para tratamento superior
      }
    } catch (error) {
      Logger.log("Erro em wtgUpdateRecordById_: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em wtgUpdateRecordById_: " + error.message);
    throw error;
  }
}
