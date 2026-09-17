/**
 * @file SchemaService.gs
 * @description Catalogo padronizado de abas e colunas para CRUD do projeto.
 * Mantem compatibilidade com montadores existentes e garante que cada entidade
 * tenha uma aba com cabecalhos minimos antes de operacoes de leitura/escrita.
 */

var PROJECT_SCHEMA_DEFINITIONS = {"AUDIT_LOGS":{"entity":"AUDIT_LOGS","sheetName":"Audit_Logs","headers":["ID","Timestamp","Level","Action","Entity","RecordID","UserID","Message","Details","CreatedAt"],"required":["ID"],"identifier":"ID","source":"baseline"},"SETTINGS":{"entity":"SETTINGS","sheetName":"Settings","headers":["Key","Value","Description","Scope","UpdatedAt","UpdatedBy"],"required":["Key"],"identifier":"Key","source":"baseline"},"USERS":{"entity":"USERS","sheetName":"Usuarios","headers":["ID","Username","Password","Role","Nome","Email","Status","LastLoginAt","CriadoEm","AtualizadoEm"],"required":["ID"],"identifier":"ID","source":"baseline"}};

// Aba real lida pelo login por senha deste projeto (ver Auth/AuthLogin/UserService).
var SYNTHETIC_USERS_LOGIN_SCHEMAS = [
  { sheetName: 'Usuarios', headers: ['ID','Username','Password','Role','Nome','Email','Status','LastLoginAt','CriadoEm','AtualizadoEm'] }
];

var SchemaService = (function() {
  function clone_(value) {
    try {
      return JSON.parse(JSON.stringify(value));
    } catch (error) {
      Logger.log("Erro em clone_: " + error.message);
      throw error;
    }
  }

  function normalizeEntityName_(entityName) {
    return String(entityName || '').trim().toUpperCase().replace(/[^A-Z0-9]+/g, '_').replace(/^_+|_+$/g, '');
  }

  function getSpreadsheet_(options) {
    try {
      options = options || {};
      if (options.spreadsheet) return options.spreadsheet;
      if (options.spreadsheetId) return SpreadsheetApp.openById(options.spreadsheetId);
      if (typeof getSpreadsheetId === 'function') {
        var idFromFunction = getSpreadsheetId();
        if (idFromFunction) return SpreadsheetApp.openById(idFromFunction);
      }
      if (typeof CONFIG !== 'undefined' && CONFIG.SPREADSHEET_ID) {
        return SpreadsheetApp.openById(CONFIG.SPREADSHEET_ID);
      }
      if (typeof Config !== 'undefined' && Config.SPREADSHEET_ID) {
        return SpreadsheetApp.openById(Config.SPREADSHEET_ID);
      }
      var idFromProperties = PropertiesService.getScriptProperties().getProperty('SPREADSHEETS_ID') || PropertiesService.getScriptProperties().getProperty('SPREADSHEET_ID');
      if (idFromProperties) return SpreadsheetApp.openById(idFromProperties);
      return getBoundSpreadsheet_();
    } catch (error) {
      Logger.log("Erro em getSpreadsheet_: " + error.message);
      throw error;
    }
  }

  // codex-schema-crud-compat: keep SchemaService aligned with runtime CRUD/config sources.
  function normalizeEntityKey_(value) {
    try {
      return String(value || '')
        .replace(/^DB_/, '')
        .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
        .replace(/[^A-Za-z0-9]+/g, '_')
        .replace(/^_+|_+$/g, '')
        .toUpperCase();
    } catch (error) {
      Logger.log("Erro em normalizeEntityKey_: " + error.message);
      throw error;
    }
  }

  function headersFromStructure_(structure) {
    if (!structure) return [];
    if (Array.isArray(structure)) return structure;
    if (Array.isArray(structure.headers)) return structure.headers;
    if (Array.isArray(structure.columns)) return structure.columns;
    return [];
  }

  function upsertRuntimeSchema_(schemas, entityName, sheetName, headers, source) {
    try {
      var cleanHeaders = (headers || []).filter(function(header, index, arr) {
        return header && arr.indexOf(header) === index;
      });
      if (!sheetName || cleanHeaders.length === 0) return;

      var key = normalizeEntityKey_(entityName || sheetName);
      var existing = schemas[key] || {};
      schemas[key] = {
        entity: existing.entity || key,
        sheetName: sheetName,
        headers: cleanHeaders,
        required: existing.required && existing.required.length ? existing.required : [cleanHeaders[0]],
        identifier: existing.identifier || cleanHeaders[0],
        source: source || existing.source || 'runtime'
      };
    } catch (error) {
      Logger.log("Erro em upsertRuntimeSchema_: " + error.message);
      throw error;
    }
  }

  function applyHeaderOverrides_(schemas) {
    try {
      if (typeof CODEX_SCHEMA_HEADER_OVERRIDES === 'undefined') return;
      Object.keys(CODEX_SCHEMA_HEADER_OVERRIDES).forEach(function(key) {
        var schema = schemas[key];
        if (!schema) return;
        upsertRuntimeSchema_(
          schemas,
          key,
          schema.sheetName,
          CODEX_SCHEMA_HEADER_OVERRIDES[key],
          'CODEX_SCHEMA_HEADER_OVERRIDES'
        );
      });
    } catch (error) {
      Logger.log('SchemaService header override compatibility skipped: ' + error.message);
    }
  }

  function addDbCoreSchemas_(schemas) {
    try {
      if (typeof DB_Core === 'undefined' || !DB_Core._SCHEMAS) return;
      Object.keys(DB_Core._SCHEMAS).forEach(function(sheetName) {
        upsertRuntimeSchema_(schemas, sheetName, sheetName, DB_Core._SCHEMAS[sheetName], 'DB_Core._SCHEMAS');
      });
    } catch (error) {
      Logger.log('SchemaService DB_Core compatibility skipped: ' + error.message);
    }
  }

  function addSheetStructureSchemas_(schemas) {
    try {
      try {
        if (typeof SHEET_STRUCTURES === 'undefined') return;
        Object.keys(SHEET_STRUCTURES).forEach(function(sheetName) {
          upsertRuntimeSchema_(
            schemas,
            sheetName,
            sheetName,
            headersFromStructure_(SHEET_STRUCTURES[sheetName]),
            'SHEET_STRUCTURES'
          );
        });
      } catch (error) {
        Logger.log('SchemaService SHEET_STRUCTURES compatibility skipped: ' + error.message);
      }
    } catch (error) {
      Logger.log("Erro em addSheetStructureSchemas_: " + error.message);
      throw error;
    }
  }

  function addConfigGetSheetSchemas_(schemas) {
    try {
      if (typeof Config_getSheet !== 'function') return;
      ['alunos', 'escolas', 'professores', 'usuarios', 'users'].forEach(function(key) {
        try {
          var definition = Config_getSheet(key);
          if (definition && definition.name && definition.headers) {
            upsertRuntimeSchema_(schemas, key, definition.name, definition.headers, 'Config_getSheet');
          }
        } catch (ignored) {}
      });
    } catch (error) {
      Logger.log('SchemaService Config_getSheet compatibility skipped: ' + error.message);
    }
  }

  function addGetSheetNamesSchemas_(schemas) {
    try {
      try {
        if (typeof getSheetNames !== 'function') return;
        var sheetNames = getSheetNames();
        var knownHeaders = {
          ACHIEVEMENTS: ['id', 'name', 'description', 'icon', 'criteria', 'points', 'active', 'created_at', 'updated_at'],
          SCIENTISTS: ['id', 'name', 'area', 'institution', 'short_description', 'full_bio', 'image_url', 'difficulty', 'display_order', 'active', 'created_at', 'updated_at'],
          QUESTS: ['id', 'title', 'description', 'status', 'objectives', 'criteria', 'reward', 'difficulty', 'display_order', 'created_at', 'updated_at'],
          ITEMS: ['id', 'name', 'description', 'type', 'rarity', 'price', 'metadata', 'active', 'created_at', 'updated_at'],
          FEEDBACK: ['id', 'user_id', 'context', 'rating', 'message', 'status', 'created_at', 'updated_at'],
          LEADERBOARD_CONFIG: ['id', 'name', 'metric', 'period', 'rules', 'active', 'created_at', 'updated_at'],
          QUEST_LOG: ['id', 'user_id', 'quest_id', 'status', 'progress', 'score', 'started_at', 'completed_at', 'created_at', 'updated_at']
        };

        Object.keys(sheetNames).forEach(function(key) {
          if (typeof sheetNames[key] !== 'string') return;
          upsertRuntimeSchema_(
            schemas,
            key,
            sheetNames[key],
            knownHeaders[key] || ['id', 'name', 'status', 'created_at', 'updated_at'],
            'getSheetNames'
          );
        });
      } catch (error) {
        Logger.log('SchemaService getSheetNames compatibility skipped: ' + error.message);
      }
    } catch (error) {
      Logger.log("Erro em addGetSheetNamesSchemas_: " + error.message);
      throw error;
    }
  }

  function collectRuntimeSchemas_() {
    var schemas = clone_(PROJECT_SCHEMA_DEFINITIONS);
    applyHeaderOverrides_(schemas);
    addDbCoreSchemas_(schemas);
    addSheetStructureSchemas_(schemas);
    addConfigGetSheetSchemas_(schemas);
    addGetSheetNamesSchemas_(schemas);
    return schemas;
  }

  function getSchema(entityName) {
    try {
      var requestedKey = normalizeEntityKey_(entityName);
      var schemas = collectRuntimeSchemas_();
      var schema = schemas[requestedKey];

      if (!schema) {
        Object.keys(schemas).some(function(key) {
          if (normalizeEntityKey_(schemas[key].sheetName) === requestedKey) {
            schema = schemas[key];
            return true;
          }
          return false;
        });
      }

      if (!schema) {
        throw new Error('Schema nao encontrado para entidade: ' + entityName);
      }
      return clone_(schema);
    } catch (error) {
      Logger.log("Erro em getSchema: " + error.message);
      throw error;
    }
  }

  function getSchemas(options) {
    try {
      options = options || {};
      var schemas = collectRuntimeSchemas_();
      if (!options.asArray) return schemas;
      return Object.keys(schemas).sort().map(function(key) { return schemas[key]; });
    } catch (error) {
      Logger.log("Erro em getSchemas: " + error.message);
      throw error;
    }
  }

  function readHeaders_(sheet) {
    try {
      try {
        if (!sheet || sheet.getLastColumn() === 0) return [];
        return sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0].map(function(header) {
          return String(header || '').trim();
        }).filter(Boolean);
      } catch (error) {
        Logger.log("Erro em readHeaders_: " + error.message);
        throw error;
      }
    } catch (error) {
      Logger.log("Erro em readHeaders_: " + error.message);
      throw error;
    }
  }

  function ensureHeaders_(sheet, headers) {
    try {
      var current = readHeaders_(sheet);
      var missing = headers.filter(function(header) { return current.indexOf(header) === -1; });
      if (current.length === 0) {
        sheet.getRange(1, 1, 1, headers.length).setValues([headers]);
        sheet.setFrozenRows(1);
        return headers;
      }
      if (missing.length > 0) {
        sheet.getRange(1, current.length + 1, 1, missing.length).setValues([missing]);
        return current.concat(missing);
      }
      return current;
    } catch (error) {
      Logger.log("Erro em ensureHeaders_: " + error.message);
      throw error; // Re-lança para tratamento superior
    }
  }

  function ensureSheet(entityName, options) {
    try {
      options = options || {};
      var schema = typeof entityName === 'object' ? entityName : getSchema(entityName);
      var spreadsheet = getSpreadsheet_(options);
      var sheet = spreadsheet.getSheetByName(schema.sheetName);
      var created = false;
      if (!sheet) {
        sheet = spreadsheet.insertSheet(schema.sheetName);
        created = true;
      }
      var headers = ensureHeaders_(sheet, schema.headers || []);
      return {
        ok: true,
        entity: schema.entity,
        sheetName: schema.sheetName,
        created: created,
        headers: headers
      };
    } catch (error) {
      Logger.log("Erro em ensureSheet: " + error.message);
      throw error;
    }
  }

  function ensureSheetForRecord(entityName, record, options) {
    try {
      var schema = getSchema(entityName);
      var extraHeaders = Object.keys(record || {}).filter(function(key) {
        return schema.headers.indexOf(key) === -1;
      });
      if (extraHeaders.length) {
        schema.headers = schema.headers.concat(extraHeaders);
      }
      return ensureSheet(schema, options);
    } catch (error) {
      Logger.log("Erro em ensureSheetForRecord: " + error.message);
      throw error;
    }
  }

  function ensureAllSheets(options) {
    try {
      options = options || {};
      var schemas = getSchemas({ asArray: true });
      var ensured = schemas.map(function(schema) {
        return ensureSheet(schema, options);
      });
      // codex-auto-seed-admins: popula os 15 admins sinteticos (idempotente; opt-out via options.seedAdminUsers === false).
      if (options.seedAdminUsers !== false) {
        try { seedSyntheticUsers(options); } catch (seedError) {}
      }
      return ensured;
    } catch (error) {
      Logger.log("Erro em ensureAllSheets: " + error.message);
      throw error;
    }
  }

  function validateData(entityName, data) {
    var schema = getSchema(entityName);
    var missing = (schema.required || []).filter(function(field) {
      return data == null || data[field] === undefined || data[field] === null || data[field] === '';
    });
    return {
      valid: missing.length === 0,
      entity: schema.entity,
      missing: missing,
      required: schema.required || [],
      headers: schema.headers || []
    };
  }

  function validateSpreadsheet(options) {
    try {
      options = options || {};
      var spreadsheet = getSpreadsheet_(options);
      return getSchemas({ asArray: true }).map(function(schema) {
        var sheet = spreadsheet.getSheetByName(schema.sheetName);
        if (!sheet) {
          return { ok: false, entity: schema.entity, sheetName: schema.sheetName, missingSheet: true, missingHeaders: schema.headers };
        }
        var headers = readHeaders_(sheet);
        var missingHeaders = schema.headers.filter(function(header) { return headers.indexOf(header) === -1; });
        return {
          ok: missingHeaders.length === 0,
          entity: schema.entity,
          sheetName: schema.sheetName,
          missingSheet: false,
          missingHeaders: missingHeaders
        };
      });
    } catch (error) {
      Logger.log("Erro em validateSpreadsheet: " + error.message);
      throw error;
    }
  }

  function getSheet(entityName, options) {
    try {
      var result = ensureSheet(entityName, options || {});
      return getSpreadsheet_(options || {}).getSheetByName(result.sheetName);
    } catch (error) {
      Logger.log("Erro em getSheet: " + error.message);
      throw error;
    }
  }

  // ---------------------------------------------------------------------------
  // Usuarios administradores sinteticos (identicos em todos os 22 webapps).
  // Cria o MESMO conjunto de 15 administradores em qualquer projeto, todos com
  // a senha 'admin123' gravada em TEXTO PURO. A insercao e feita via
  // createRecord: usa o createRecord nativo do projeto quando existir; caso
  // contrario usa um createRecord embutido que grava diretamente na aba.
  // ---------------------------------------------------------------------------
  var SYNTHETIC_ADMIN_USERS_PASSWORD = 'admin123';
  var SYNTHETIC_ADMIN_USERS_PREFIX = 'SYN-ADMIN-';
  var SYNTHETIC_ADMIN_USERS_SEED = [
    'Ana Beatriz Carvalho',
    'Bruno Henrique Alves',
    'Carla Regina Souza',
    'Diego Martins Lima',
    'Eduarda Nunes Pereira',
    'Felipe Augusto Rocha',
    'Gabriela Santos Dias',
    'Henrique Oliveira Costa',
    'Isabela Fernandes Melo',
    'Joao Pedro Ribeiro',
    'Larissa Gomes Barros',
    'Marcelo Tavares Pinto',
    'Natalia Cardoso Freitas',
    'Otavio Ramos Teixeira',
    'Patricia Lopes Moreira'
  ];

  function buildSyntheticAdminUsers_() {
    try {
      return SYNTHETIC_ADMIN_USERS_SEED.map(function(name, index) {
        var n = index + 1;
        var pad = (n < 10 ? '0' : '') + n;
        return {
          id: SYNTHETIC_ADMIN_USERS_PREFIX + pad,
          name: name,
          email: 'admin' + pad + '@synthetic.local',
          username: 'admin' + pad,
          role: 'admin',
          status: 'ativo',
          password: SYNTHETIC_ADMIN_USERS_PASSWORD
        };
      });
    } catch (error) {
      Logger.log("Erro em buildSyntheticAdminUsers_: " + error.message);
      throw error;
    }
  }

  function isUserEntitySchema_(schema) {
    try {
      var headers = (schema.headers || []).map(function(header) { return String(header || '').toLowerCase(); });
      var hasSecret = headers.some(function(header) {
        return header.indexOf('password') >= 0 || header.indexOf('senha') >= 0 || header === 'hash';
      });
      var hasIdentity = headers.some(function(header) {
        return header.indexOf('email') >= 0 || header.indexOf('username') >= 0 || header.indexOf('usuario') >= 0 || header.indexOf('login') >= 0;
      });
      return hasSecret && hasIdentity;
    } catch (error) {
      Logger.log("Erro em isUserEntitySchema_: " + error.message);
      throw error;
    }
  }

  function syntheticAdminUserValue_(header, user, nowStr) {
    try {
      var n = String(header || '').toLowerCase().replace(/[^a-z0-9]+/g, '');
      if (!n) return '';
      if (n.indexOf('password') >= 0 || n.indexOf('senha') >= 0 || n === 'hash') return user.password;
      if (n.indexOf('salt') >= 0) return '';
      if (n.indexOf('lastlogin') >= 0 || n.indexOf('ultimo') >= 0) return '';
      if (n.indexOf('username') >= 0 || n.indexOf('usuario') >= 0 || n.indexOf('login') >= 0) return user.username;
      if (n === 'id' || n.indexOf('recordid') >= 0 || n.indexOf('userid') >= 0 || n.indexOf('codigo') >= 0 || n === 'key') return user.id;
      if (n.indexOf('email') >= 0) return user.email;
      if (n.indexOf('fullname') >= 0 || n.indexOf('name') >= 0 || n.indexOf('nome') >= 0) return user.name;
      if (n.indexOf('role') >= 0 || n.indexOf('perfil') >= 0 || n.indexOf('papel') >= 0 || n.indexOf('nivel') >= 0) return user.role;
      if (n.indexOf('status') >= 0 || n.indexOf('situacao') >= 0 || n.indexOf('ativo') >= 0 || n.indexOf('active') >= 0) return user.status;
      if (n.indexOf('updated') >= 0 || n.indexOf('atualizado') >= 0) return nowStr;
      if (n.indexOf('created') >= 0 || n.indexOf('criado') >= 0 || n.indexOf('timestamp') >= 0 || n.indexOf('data') >= 0) return nowStr;
      if (n.indexOf('description') >= 0 || n.indexOf('descricao') >= 0 || n.indexOf('notes') >= 0 || n.indexOf('observ') >= 0 || n.indexOf('message') >= 0) return 'Usuario administrador sintetico';
      if (n.indexOf('phone') >= 0 || n.indexOf('telefone') >= 0) return '';
      return '';
    } catch (error) {
      Logger.log("Erro em syntheticAdminUserValue_: " + error.message);
      throw error;
    }
  }

  function buildUserRecord_(user, headers, nowStr) {
    try {
      var record = {};
      headers.forEach(function(header) {
        record[header] = syntheticAdminUserValue_(header, user, nowStr);
      });
      return record;
    } catch (error) {
      Logger.log("Erro em buildUserRecord_: " + error.message);
      throw error;
    }
  }

  function nativeCreateRecordAvailable_() {
    try { return typeof createRecord === 'function'; } catch (error) { return false; }
  }

  // createRecord embutido: usa o createRecord nativo do projeto quando existir
  // (tentando entidade e depois nome da aba); caso contrario grava direto na aba.
  function seedCreateRecord_(entity, schema, headers, sheet, record) {
    if (nativeCreateRecordAvailable_()) {
      var attempts = [entity, schema.sheetName];
      for (var a = 0; a < attempts.length; a++) {
        try {
          var result = createRecord(attempts[a], record);
          if (!result || result.success !== false) return 'native';
        } catch (ignored) {}
      }
    }
    sheet.appendRow(headers.map(function(header) {
      return record[header] !== undefined ? record[header] : '';
    }));
    return 'embedded';
  }

  function existingUserKeys_(sheet, headers) {
    try {
      var keys = {};
      if (!sheet || sheet.getLastRow() < 2) return keys;
      var idxId = -1, idxEmail = -1, idxUser = -1;
      headers.forEach(function(header, i) {
        var h = String(header || '').toLowerCase();
        if (idxId === -1 && (h === 'id' || h.indexOf('recordid') >= 0)) idxId = i;
        if (idxEmail === -1 && h.indexOf('email') >= 0) idxEmail = i;
        if (idxUser === -1 && (h.indexOf('username') >= 0 || h.indexOf('usuario') >= 0 || h.indexOf('login') >= 0)) idxUser = i;
      });
      var values = sheet.getRange(2, 1, sheet.getLastRow() - 1, headers.length).getValues();
      values.forEach(function(row) {
        [idxId, idxEmail, idxUser].forEach(function(idx) {
          if (idx >= 0) {
            var value = String(row[idx] || '').trim().toLowerCase();
            if (value) keys[value] = true;
          }
        });
      });
      return keys;
    } catch (error) {
      Logger.log("Erro em existingUserKeys_: " + error.message);
      throw error;
    }
  }

  function isSyntheticAdminRow_(row) {
    try {
      for (var i = 0; i < row.length; i++) {
        if (String(row[i] || '').indexOf(SYNTHETIC_ADMIN_USERS_PREFIX) === 0) return true;
      }
      return false;
    } catch (error) {
      Logger.log("Erro em isSyntheticAdminRow_: " + error.message);
      throw error;
    }
  }

  function getSyntheticAdminUsers() {
    try {
      return buildSyntheticAdminUsers_().map(function(user) {
        return {
          id: user.id,
          name: user.name,
          email: user.email,
          username: user.username,
          role: user.role,
          status: user.status,
          password: user.password
        };
      });
    } catch (error) {
      Logger.log("Erro em getSyntheticAdminUsers: " + error.message);
      throw error;
    }
  }

  // Reune as abas de usuario do catalogo + as abas reais de login declaradas
  // pelo projeto em SYNTHETIC_USERS_LOGIN_SCHEMAS. Estas tem prioridade e sao
  // gravadas exatamente com os cabecalhos/ordem que o login do projeto espera.
  function syntheticUserTargets_() {
    var loginSchemas = [];
    try {
      if (typeof SYNTHETIC_USERS_LOGIN_SCHEMAS !== 'undefined' && SYNTHETIC_USERS_LOGIN_SCHEMAS) {
        loginSchemas = SYNTHETIC_USERS_LOGIN_SCHEMAS;
      }
    } catch (ignored) {}
    // Se o projeto declarou SYNTHETIC_USERS_LOGIN_SCHEMAS explicitamente, usa
    // SOMENTE eles — evita duplicatas com entidades "Users" do catálogo baseline.
    if (loginSchemas.length > 0) {
      return loginSchemas.map(function(ls) {
        return {
          entity: ls.entity || ls.sheetName,
          sheetName: ls.sheetName,
          headers: ls.headers || [],
          required: [(ls.headers && ls.headers[0]) || 'ID'],
          identifier: (ls.headers && ls.headers[0]) || 'ID',
          constants: ls.constants || {},
          login: true
        };
      });
    }
    return getSchemas({ asArray: true }).filter(isUserEntitySchema_);
  }

  function clearSyntheticUsers(options) {
    try {
      try {
        try {
          options = options || {};
          var summary = [];
          syntheticUserTargets_().forEach(function(schema) {
            var sheet = getSpreadsheet_(options).getSheetByName(schema.sheetName);
            if (!sheet || sheet.getLastRow() < 2) {
              summary.push({ entity: schema.entity, sheetName: schema.sheetName, removed: 0 });
              return;
            }
            var lastColumn = sheet.getLastColumn();
            var values = sheet.getRange(2, 1, sheet.getLastRow() - 1, lastColumn).getValues();
            var keep = values.filter(function(row) { return !isSyntheticAdminRow_(row); });
            var removed = values.length - keep.length;
            if (removed > 0) {
              sheet.getRange(2, 1, values.length, lastColumn).clearContent();
              if (keep.length > 0) {
                sheet.getRange(2, 1, keep.length, lastColumn).setValues(keep);
              }
            }
            summary.push({ entity: schema.entity, sheetName: schema.sheetName, removed: removed });
          });
          return { ok: true, action: 'clear-synthetic-users', sheets: summary };
        } catch (error) {
          Logger.log("Erro em clearSyntheticUsers: " + error.message);
          throw error; // Re-lança para tratamento superior
        }
      } catch (error) {
        Logger.log("Erro em clearSyntheticUsers: " + error.message);
        throw error;
      }
    } catch (error) {
      Logger.log("Erro em clearSyntheticUsers: " + error.message);
      throw error;
    }
  }

  function seedSyntheticUsers(options) {
    try {
      try {
        try {
          options = options || {};
          if (options.reset === true) {
            clearSyntheticUsers(options);
          }
          var nowStr = Utilities.formatDate(new Date(), Session.getScriptTimeZone(), 'yyyy-MM-dd HH:mm:ss');
          var users = buildSyntheticAdminUsers_();
          var summary = [];
          syntheticUserTargets_().forEach(function(schema) {
            ensureSheet(schema, options);
            var sheet = getSpreadsheet_(options).getSheetByName(schema.sheetName);
            var headers = readHeaders_(sheet);
            var constants = schema.constants || {};
            var inserted = 0, skipped = 0, via = null;
            if (headers.length) {
              var existing = existingUserKeys_(sheet, headers);
              users.forEach(function(user) {
                var duplicate = existing[user.id.toLowerCase()] || existing[user.email.toLowerCase()] || existing[user.username.toLowerCase()];
                if (duplicate) { skipped++; return; }
                var record = buildUserRecord_(user, headers, nowStr);
                headers.forEach(function(h) { if (constants[h] !== undefined) record[h] = constants[h]; });
                if (schema.login) {
                  sheet.appendRow(headers.map(function(h) { return record[h] !== undefined ? record[h] : ''; }));
                  via = 'login-sheet';
                } else {
                  via = seedCreateRecord_(schema.entity, schema, headers, sheet, record);
                }
                inserted++;
              });
            }
            summary.push({ entity: schema.entity, sheetName: schema.sheetName, inserted: inserted, skipped: skipped, via: via });
          });
          return {
            ok: true,
            action: 'seed-synthetic-users',
            password: SYNTHETIC_ADMIN_USERS_PASSWORD,
            totalUsers: users.length,
            sheets: summary
          };
        } catch (error) {
          Logger.log("Erro em seedSyntheticUsers: " + error.message);
          throw error; // Re-lança para tratamento superior
        }
      } catch (error) {
        Logger.log("Erro em seedSyntheticUsers: " + error.message);
        throw error;
      }
    } catch (error) {
      Logger.log("Erro em seedSyntheticUsers: " + error.message);
      throw error;
    }
  }

  return {
    getSchemas: getSchemas,
    getSchema: getSchema,
    getSheet: getSheet,
    ensureSheet: ensureSheet,
    ensureSheetForRecord: ensureSheetForRecord,
    ensureAllSheets: ensureAllSheets,
    validateData: validateData,
    validateSpreadsheet: validateSpreadsheet,
    seedSyntheticVisualizationData: seedSyntheticVisualizationData,
    seedSyntheticUsers: seedSyntheticUsers,
    clearSyntheticUsers: clearSyntheticUsers
  };
})();

function seedSyntheticAdminUsers(options) {
  return SchemaService.seedSyntheticUsers(options || {});
}

function clearSyntheticAdminUsers(options) {
  return SchemaService.clearSyntheticUsers(options || {});
}

function listSyntheticAdminUsers() {
  return SchemaService.getSyntheticAdminUsers();
}

function runSchemaServiceSetup(options) {
  var sheets = SchemaService.ensureAllSheets(options || {});
  // Bootstrap da gamificação: registra os distintivos padrão de trânsito/cidadania
  // (idempotente) para que a engine funcione já no primeiro uso.
  var trafficBadges = null;
  if (typeof seedTrafficBadges === 'function') {
    try { trafficBadges = seedTrafficBadges().data; } catch (e) {}
  }
  return { sheets: sheets, trafficBadges: trafficBadges };
}

function runSchemaServiceValidation(options) {
  return SchemaService.validateSpreadsheet(options || {});
}

/**
 * Migracao unica e idempotente da formacao de planilha deste projeto.
 * Consolida os usuarios na aba canonica 'Usuarios' (cabecalho enriquecido com as
 * colunas migradas de 'Users') e aposenta as abas legadas:
 *   - 'Users' (catalogo baseline duplicado): linhas mapeadas e movidas p/ 'Usuarios'.
 *   - 'YOUR_SPREADSHEET_ID_HERE' (lixo de um scanner que leu o placeholder): renomeada.
 * Nao apaga nada — renomeia as abas antigas para 'ZZ_Obsoleto_*' para revisao manual.
 * Observacao: 'PasswordHash' legado e migrado para 'Password'; como hash nao e
 * reversivel, esses usuarios podem precisar de redefinicao de senha.
 * Rode UMA vez pelo editor do Apps Script apos o deploy.
 */
function migrateLegacyUserSheets() {
  try {
    try {
      var ss = (typeof getSpreadsheet_ === 'function')
        ? getSpreadsheet_()
        : SpreadsheetApp.openById(getSpreadsheetId());
      var stamp = Utilities.formatDate(new Date(), Session.getScriptTimeZone(), 'yyyyMMdd');
      var summary = { target: 'Usuarios', movedFromUsers: 0, skipped: 0, usersTab: 'ausente', placeholderTab: 'ausente' };

      // 1) Garante a aba canonica 'Usuarios' com o cabecalho atual.
      SchemaService.ensureSheet('USERS', { spreadsheet: ss });
      var dest = ss.getSheetByName('Usuarios');
      var destHeaders = dest.getRange(1, 1, 1, dest.getLastColumn()).getValues()[0]
        .map(function (h) { return String(h || '').trim(); });

      var COLMAP = {
        'ID': 'ID', 'Username': 'Username', 'Name': 'Nome', 'Nome': 'Nome',
        'Email': 'Email', 'PasswordHash': 'Password', 'Password': 'Password', 'Senha': 'Password',
        'Role': 'Role', 'Status': 'Status', 'LastLoginAt': 'LastLoginAt',
        'CreatedAt': 'CriadoEm', 'CriadoEm': 'CriadoEm', 'UpdatedAt': 'AtualizadoEm', 'AtualizadoEm': 'AtualizadoEm'
      };

      function destKeys_() {
        try {
          var keys = {};
          if (dest.getLastRow() < 2) return keys;
          var vals = dest.getRange(2, 1, dest.getLastRow() - 1, destHeaders.length).getValues();
          var iU = destHeaders.indexOf('Username'), iE = destHeaders.indexOf('Email'), iI = destHeaders.indexOf('ID');
          vals.forEach(function (r) {
            [iU, iE, iI].forEach(function (ix) {
              if (ix >= 0) { var v = String(r[ix] || '').trim().toLowerCase(); if (v) keys[v] = true; }
            });
          });
          return keys;
        } catch (error) {
          Logger.log("Erro em destKeys_: " + error.message);
          throw error; // Re-lança para tratamento superior
        }
      }

      // 2) Migra linhas da aba legada 'Users', se houver dados.
      var src = ss.getSheetByName('Users');
      if (src && src.getLastRow() > 1) {
        var sVals = src.getDataRange().getValues();
        var sHead = sVals[0].map(function (h) { return String(h || '').trim(); });
        var keys = destKeys_();
        for (var r = 1; r < sVals.length; r++) {
          var row = sVals[r];
          if (!row.some(function (c) { return c !== '' && c !== null; })) continue;
          var rec = {};
          sHead.forEach(function (h, i) { var canon = COLMAP[h]; if (canon && rec[canon] === undefined) rec[canon] = row[i]; });
          var dk = String(rec.Username || rec.Email || rec.ID || '').trim().toLowerCase();
          if (!dk || keys[dk]) { summary.skipped++; continue; }
          dest.appendRow(destHeaders.map(function (h) { return rec[h] !== undefined ? rec[h] : ''; }));
          keys[dk] = true; summary.movedFromUsers++;
        }
      }
      if (src) { src.setName('ZZ_Obsoleto_Users_' + stamp); summary.usersTab = src.getName(); }

      // 3) Aposenta a aba-lixo do placeholder (nunca foi entidade real).
      var ph = ss.getSheetByName('YOUR_SPREADSHEET_ID_HERE');
      if (ph) { ph.setName('ZZ_Obsoleto_Placeholder_' + stamp); summary.placeholderTab = ph.getName(); }

      return summary;
    } catch (error) {
      Logger.log("Erro em migrateLegacyUserSheets: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em migrateLegacyUserSheets: " + error.message);
    throw error;
  }
}

/* FLEET_AI_SCHEMA_V2: estrutura real + fixtures semanticos explicitamente catalogados. */
function restoreAiCrudStructure(options) {
  options = options || {};
  if (typeof SchemaService === 'undefined') throw new Error('SchemaService indisponivel.');
  if (typeof SchemaService.ensureAllSheets === 'function') return SchemaService.ensureAllSheets(options);
  if (typeof SchemaService.ensureAll === 'function') return SchemaService.ensureAll(options);
  if (typeof SchemaService.restore === 'function') return SchemaService.restore(options);
  throw new Error('SchemaService nao expoe restauracao de estrutura.');
}

function seedSyntheticAiData(options) {
  try {
    try {
      options = options || {};
      var ss = options.spreadsheet || getBoundSpreadsheet_();
      var catalog = typeof AI_FIXTURE_CATALOG !== 'undefined' ? AI_FIXTURE_CATALOG : [];
      var summary = [];
      catalog.forEach(function(fixture) {
        var sheet = ss.getSheetByName(fixture.sheetName);
        if (!sheet) sheet = ss.insertSheet(fixture.sheetName);
        var currentHeaders = sheet.getLastColumn() ? sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0] : [];
        if (!currentHeaders.length || currentHeaders.every(function(value) { return !value; })) {
          sheet.getRange(1, 1, 1, fixture.headers.length).setValues([fixture.headers]);
          currentHeaders = fixture.headers.slice();
        }
        var sameHeaders = fixture.headers.length === currentHeaders.length &&
          fixture.headers.every(function(header, index) { return String(currentHeaders[index]) === String(header); });
        if (!sameHeaders) throw new Error('Fixture incompatível com os cabecalhos de ' + fixture.sheetName);
        var inserted = 0;
        if (options.reset === true && sheet.getLastRow() > 1) {
          sheet.getRange(2, 1, sheet.getLastRow() - 1, sheet.getLastColumn()).clearContent();
        }
        if (sheet.getLastRow() < 2 || options.append === true) {
          sheet.getRange(sheet.getLastRow() + 1, 1, fixture.rows.length, fixture.headers.length).setValues(fixture.rows);
          inserted = fixture.rows.length;
        }
        summary.push({ sheetName: fixture.sheetName, purpose: fixture.purpose, inserted: inserted });
      });
      return { ok: true, action: 'seed-semantic-ai-fixtures', entities: summary };
    } catch (error) {
      Logger.log("Erro em seedSyntheticAiData: " + error.message);
      throw error; // Re-lança para tratamento superior
    }
  } catch (error) {
    Logger.log("Erro em seedSyntheticAiData: " + error.message);
    throw error;
  }
}

function prepareAiIntegrationFixtures(options) {
  options = options || {};
  return {
    structure: restoreAiCrudStructure(options),
    synthetic: seedSyntheticAiData(options),
    visualization: seedSyntheticVisualizationData(options)
  };
}
/* CODEX_VISUALIZATION_FIXTURES_V1: dados pequenos para validar os notebook.py. */
function _codexVisualizationSpreadsheet_(options) {
  try {
    try {
      options = options || {};
      if (options.spreadsheet) return options.spreadsheet;
      if (typeof getBoundSpreadsheet_ === 'function') return getBoundSpreadsheet_();
      if (typeof getSS === 'function') return getSS();
      var active = SpreadsheetApp.getActiveSpreadsheet();
      if (active) return active;
      var props = PropertiesService.getScriptProperties();
      var spreadsheetId = props.getProperty('SPREADSHEETS_ID') || props.getProperty('SPREADSHEET_ID');
      if (!spreadsheetId) throw new Error('SPREADSHEETS_ID nao configurado para seed de visualizacoes.');
      return SpreadsheetApp.openById(spreadsheetId);
    } catch (error) {
      Logger.log("Erro em _codexVisualizationSpreadsheet_: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em _codexVisualizationSpreadsheet_: " + error.message);
    throw error;
  }
}

function _codexVisualizationRows_(projectName) {
  try {
    var today = new Date();
    var categories = ['Leitura', 'Matematica', 'Ciencias', 'Artes'];
    var statuses = ['Planejado', 'Em andamento', 'Concluido'];
    var rows = [];
    for (var i = 0; i < 16; i++) {
      var d = new Date(today.getFullYear(), today.getMonth(), today.getDate() - (15 - i));
      var categoria = categories[i % categories.length];
      var status = statuses[i % statuses.length];
      var ciclo = Math.floor(i / categories.length);
      var metricaA = 48 + (i * 3) + (ciclo * 2);
      var metricaB = 32 + ((i % 5) * 7) + ciclo;
      var valor = 120 + (i * 11) + ((i % 3) * 17);
      rows.push([
        'VIS_SYN_' + Utilities.formatString('%02d', i + 1),
        Utilities.formatDate(d, Session.getScriptTimeZone(), 'yyyy-MM-dd'),
        categoria,
        status,
        metricaA,
        metricaB,
        valor,
        'Amostra sintetica para validar graficos do notebook.py',
        projectName
      ]);
    }
    return rows;
  } catch (error) {
    Logger.log("Erro em _codexVisualizationRows_: " + error.message);
    throw error;
  }
}

function seedSyntheticVisualizationData(options) {
  try {
    try {
      try {
        options = options || {};
        var ss = _codexVisualizationSpreadsheet_(options);
        var sheetName = options.sheetName || 'Visualizacoes_Sinteticas';
        var projectName = options.projectName || (typeof PROJECT_NAME !== 'undefined' ? PROJECT_NAME : ss.getName());
        var headers = ['ID', 'Data', 'Categoria', 'Status', 'Metrica_A', 'Metrica_B', 'Valor', 'Observacao', 'Projeto'];
        var rows = _codexVisualizationRows_(projectName);
        var sheet = ss.getSheetByName(sheetName) || ss.insertSheet(sheetName);

        if (sheet.getLastRow() === 0) {
          sheet.getRange(1, 1, 1, headers.length).setValues([headers]);
        } else {
          sheet.getRange(1, 1, 1, headers.length).setValues([headers]);
        }

        if (options.reset === true && sheet.getLastRow() > 1) {
          sheet.getRange(2, 1, sheet.getLastRow() - 1, Math.max(sheet.getLastColumn(), headers.length)).clearContent();
        }

        var existing = {};
        if (sheet.getLastRow() > 1) {
          sheet.getRange(2, 1, sheet.getLastRow() - 1, 1).getValues().forEach(function(row) {
            existing[String(row[0] || '')] = true;
          });
        }

        var toInsert = rows.filter(function(row) { return options.append === true || !existing[row[0]]; });
        if (toInsert.length) {
          sheet.getRange(sheet.getLastRow() + 1, 1, toInsert.length, headers.length).setValues(toInsert);
        }

        return {
          ok: true,
          action: 'seed-synthetic-visualization-data',
          sheetName: sheetName,
          inserted: toInsert.length,
          totalRows: sheet.getLastRow() - 1
        };
      } catch (error) {
        Logger.log("Erro em seedSyntheticVisualizationData: " + error.message);
        throw error; // Re-lança para tratamento superior
      }
    } catch (error) {
      Logger.log("Erro em seedSyntheticVisualizationData: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em seedSyntheticVisualizationData: " + error.message);
    throw error;
  }
}




