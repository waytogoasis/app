// DataValidationRules.gs
//
// Funcionalidade Principal: Define e aplica regras de validação de dados para as planilhas.
//
// Descrição: Este script centraliza as regras de validação de dados que devem ser aplicadas
//            às diferentes abas da Google Planilha. Isso garante a integridade e consistência
//            dos dados inseridos, seja manualmente ou via script.
//
// Integrações:
// - Google Planilha: Aplica as regras de validação diretamente nas abas.
// - SpreadsheetUtils.gs: Utiliza para interagir com as planilhas.
// - ValidationUtils.gs: Pode usar as funções de validação para verificar dados antes de aplicar regras.
//
// Funções Principais:
// - `applyValidationRulesToSheet(sheetName)`: Aplica um conjunto de regras de validação a uma aba.
// - `defineUserValidationRules()`: Define regras para a aba de usuários.
// - `defineAlunoValidationRules()`: Define regras para a aba de alunos.
//
// Observações: A validação de dados é uma camada importante para a qualidade dos dados.

/**
 * Aplica regras de validação de dados a uma aba específica.
 * @param {string} sheetName - Nome da aba para aplicar validação
 * @param {Object} [options] - Opções de validação
 * @return {Object} Resultado da operação
 */
function applyValidationRulesToSheet(sheetName, options) {
  try {
    if (!sheetName) {
      throw new Error('Nome da aba é obrigatório');
    }

    options = options || {};
    var sheet = getSheet_(sheetName);
    
    if (!sheet) {
      throw new Error('Aba não encontrada: ' + sheetName);
    }

    var rulesApplied = 0;

    // Aplica regras específicas por tipo de aba
    switch (sheetName) {
      case 'Usuarios':
      case 'Users':
        rulesApplied = defineUserValidationRules(sheet);
        break;
      
      case 'Alunos':
        rulesApplied = defineAlunoValidationRules(sheet);
        break;
      
      case 'Simulacoes':
        rulesApplied = defineSimulacaoValidationRules_(sheet);
        break;
      
      case 'Pontuacoes':
        rulesApplied = definePontuacaoValidationRules_(sheet);
        break;
      
      case 'Settings':
      case 'Config':
        rulesApplied = defineConfigValidationRules_(sheet);
        break;
      
      default:
        Logger.log('Nenhuma regra de validação definida para: ' + sheetName);
        return {
          success: true,
          rulesApplied: 0,
          message: 'Nenhuma regra definida para esta aba'
        };
    }

    Logger.log('Regras de validação aplicadas em ' + sheetName + ': ' + rulesApplied);

    return {
      success: true,
      rulesApplied: rulesApplied,
      sheetName: sheetName,
      timestamp: new Date().toISOString()
    };
  } catch (error) {
    Logger.log("Erro em applyValidationRulesToSheet: " + error.message);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Define regras de validação para a aba de usuários.
 * @param {Sheet} [sheet] - Objeto Sheet (opcional, busca se não fornecido)
 * @return {number} Número de regras aplicadas
 */
function defineUserValidationRules(sheet) {
  try {
    if (!sheet) {
      sheet = getSheet_('Usuarios');
    }

    var rulesApplied = 0;
    var lastRow = sheet.getLastRow();
    
    if (lastRow < 2) {
      return 0; // Sem dados para validar
    }

    // Assumindo estrutura: ID | Username | Password | Role | Nome | Email | Status | ...
    // Encontra índices de colunas
    var headers = sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0];
    var roleCol = headers.indexOf('Role') + 1;
    var statusCol = headers.indexOf('Status') + 1;
    var emailCol = headers.indexOf('Email') + 1;

    // Validação: Role (lista de valores permitidos)
    if (roleCol > 0) {
      var roleValues = ['admin', 'professor', 'aluno', 'user', 'teacher', 'student'];
      var roleRule = SpreadsheetApp.newDataValidation()
        .requireValueInList(roleValues, true)
        .setAllowInvalid(false)
        .setHelpText('Selecione um papel válido: ' + roleValues.join(', '))
        .build();
      
      sheet.getRange(2, roleCol, lastRow - 1, 1).setDataValidation(roleRule);
      rulesApplied++;
    }

    // Validação: Status (lista de valores permitidos)
    if (statusCol > 0) {
      var statusValues = ['ativo', 'inativo', 'suspenso'];
      var statusRule = SpreadsheetApp.newDataValidation()
        .requireValueInList(statusValues, true)
        .setAllowInvalid(false)
        .setHelpText('Selecione um status válido: ' + statusValues.join(', '))
        .build();
      
      sheet.getRange(2, statusCol, lastRow - 1, 1).setDataValidation(statusRule);
      rulesApplied++;
    }

    // Validação: Email (formato de email)
    if (emailCol > 0) {
      var emailRule = SpreadsheetApp.newDataValidation()
        .requireTextIsEmail()
        .setAllowInvalid(true) // Permite vazio
        .setHelpText('Digite um endereço de email válido')
        .build();
      
      sheet.getRange(2, emailCol, lastRow - 1, 1).setDataValidation(emailRule);
      rulesApplied++;
    }

    return rulesApplied;
  } catch (error) {
    Logger.log("Erro em defineUserValidationRules: " + error.message);
    return 0;
  }
}

/**
 * Define regras de validação para a aba de alunos.
 * @param {Sheet} [sheet] - Objeto Sheet (opcional)
 * @return {number} Número de regras aplicadas
 */
function defineAlunoValidationRules(sheet) {
  try {
    if (!sheet) {
      sheet = getSheet_('Alunos');
    }

    var rulesApplied = 0;
    var lastRow = sheet.getLastRow();
    
    if (lastRow < 2) {
      return 0;
    }

    // Encontra índices de colunas
    var headers = sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0];
    var statusCol = headers.indexOf('Status') + 1;
    var nomeCol = headers.indexOf('Nome') + 1;

    // Validação: Status
    if (statusCol > 0) {
      var statusValues = ['ativo', 'inativo', 'transferido'];
      var statusRule = SpreadsheetApp.newDataValidation()
        .requireValueInList(statusValues, true)
        .setAllowInvalid(false)
        .setHelpText('Selecione um status válido: ' + statusValues.join(', '))
        .build();
      
      sheet.getRange(2, statusCol, lastRow - 1, 1).setDataValidation(statusRule);
      rulesApplied++;
    }

    // Validação: Nome não vazio
    if (nomeCol > 0) {
      var nomeRule = SpreadsheetApp.newDataValidation()
        .requireTextLength(1, 100)
        .setAllowInvalid(false)
        .setHelpText('Nome é obrigatório (1-100 caracteres)')
        .build();
      
      sheet.getRange(2, nomeCol, lastRow - 1, 1).setDataValidation(nomeRule);
      rulesApplied++;
    }

    return rulesApplied;
  } catch (error) {
    Logger.log("Erro em defineAlunoValidationRules: " + error.message);
    return 0;
  }
}

/**
 * Define regras de validação para a aba de simulações.
 * @param {Sheet} sheet - Objeto Sheet
 * @return {number} Número de regras aplicadas
 */
function defineSimulacaoValidationRules_(sheet) {
  try {
    var rulesApplied = 0;
    var lastRow = sheet.getLastRow();
    
    if (lastRow < 2) {
      return 0;
    }

    var headers = sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0];
    var statusCol = headers.indexOf('Status') + 1;
    var tipoCol = headers.indexOf('Tipo') + 1;

    // Validação: Status
    if (statusCol > 0) {
      var statusValues = ['em_andamento', 'finalizada', 'cancelada'];
      var statusRule = SpreadsheetApp.newDataValidation()
        .requireValueInList(statusValues, true)
        .setAllowInvalid(false)
        .setHelpText('Status: ' + statusValues.join(', '))
        .build();
      
      sheet.getRange(2, statusCol, lastRow - 1, 1).setDataValidation(statusRule);
      rulesApplied++;
    }

    // Validação: Tipo
    if (tipoCol > 0) {
      var tipoValues = ['padrao', 'avancada', 'teste', 'pratica'];
      var tipoRule = SpreadsheetApp.newDataValidation()
        .requireValueInList(tipoValues, true)
        .setAllowInvalid(true)
        .setHelpText('Tipo de simulação: ' + tipoValues.join(', '))
        .build();
      
      sheet.getRange(2, tipoCol, lastRow - 1, 1).setDataValidation(tipoRule);
      rulesApplied++;
    }

    return rulesApplied;
  } catch (error) {
    Logger.log("Erro em defineSimulacaoValidationRules_: " + error.message);
    return 0;
  }
}

/**
 * Define regras de validação para a aba de pontuações.
 * @param {Sheet} sheet - Objeto Sheet
 * @return {number} Número de regras aplicadas
 */
function definePontuacaoValidationRules_(sheet) {
  try {
    var rulesApplied = 0;
    var lastRow = sheet.getLastRow();
    
    if (lastRow < 2) {
      return 0;
    }

    var headers = sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0];
    var totalCol = headers.indexOf('Total') + 1;

    // Validação: Total (número entre 0 e 100)
    if (totalCol > 0) {
      var totalRule = SpreadsheetApp.newDataValidation()
        .requireNumberBetween(0, 100)
        .setAllowInvalid(false)
        .setHelpText('Pontuação deve estar entre 0 e 100')
        .build();
      
      sheet.getRange(2, totalCol, lastRow - 1, 1).setDataValidation(totalRule);
      rulesApplied++;
    }

    return rulesApplied;
  } catch (error) {
    Logger.log("Erro em definePontuacaoValidationRules_: " + error.message);
    return 0;
  }
}

/**
 * Define regras de validação para a aba de configurações.
 * @param {Sheet} sheet - Objeto Sheet
 * @return {number} Número de regras aplicadas
 */
function defineConfigValidationRules_(sheet) {
  try {
    var rulesApplied = 0;
    var lastRow = sheet.getLastRow();
    
    if (lastRow < 2) {
      return 0;
    }

    var headers = sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0];
    var keyCol = headers.indexOf('Key') + 1;

    // Validação: Key não vazio
    if (keyCol > 0) {
      var keyRule = SpreadsheetApp.newDataValidation()
        .requireTextLength(1, 50)
        .setAllowInvalid(false)
        .setHelpText('Chave de configuração é obrigatória')
        .build();
      
      sheet.getRange(2, keyCol, lastRow - 1, 1).setDataValidation(keyRule);
      rulesApplied++;
    }

    return rulesApplied;
  } catch (error) {
    Logger.log("Erro em defineConfigValidationRules_: " + error.message);
    return 0;
  }
}

/**
 * Aplica validação a todas as abas principais do sistema.
 * @return {Object} Resultado da operação com contagem por aba
 */
function applyAllValidationRules() {
  try {
    var sheets = ['Usuarios', 'Alunos', 'Simulacoes', 'Pontuacoes', 'Settings'];
    var results = {};
    var totalRules = 0;

    sheets.forEach(function(sheetName) {
      try {
        var result = applyValidationRulesToSheet(sheetName);
        results[sheetName] = result.rulesApplied || 0;
        totalRules += results[sheetName];
      } catch (e) {
        Logger.log("Erro ao aplicar validação em " + sheetName + ": " + e.message);
        results[sheetName] = 0;
      }
    });

    return {
      success: true,
      totalRules: totalRules,
      bySheet: results,
      timestamp: new Date().toISOString()
    };
  } catch (error) {
    Logger.log("Erro em applyAllValidationRules: " + error.message);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Remove todas as validações de uma aba.
 * @param {string} sheetName - Nome da aba
 * @return {Object} Resultado da operação
 */
function clearValidationRules(sheetName) {
  try {
    var sheet = getSheet_(sheetName);
    if (!sheet) {
      throw new Error('Aba não encontrada: ' + sheetName);
    }

    var range = sheet.getDataRange();
    range.clearDataValidations();

    return {
      success: true,
      message: 'Validações removidas de ' + sheetName
    };
  } catch (error) {
    Logger.log("Erro em clearValidationRules: " + error.message);
    return {
      success: false,
      error: error.message
    };
  }
}
