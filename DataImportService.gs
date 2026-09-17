// DataImportService.gs
//
// Funcionalidade Principal: Importa dados para o sistema a partir de fontes externas.
//
// Descrição: Este script fornece funções para importar dados de alunos, simulações ou pontuações
//            a partir de arquivos CSV, planilhas externas ou outras fontes. Ele é útil para
//            preencher o sistema com dados iniciais ou para integrar com outros sistemas.
//
// Integrações:
// - Google Planilha: Destino dos dados importados.
// - SpreadsheetUtils.gs: Para acessar e manipular dados da planilha.
// - ValidationUtils.gs: Para validar os dados antes da importação.
//
// Funções Principais:
// - `importFromCsv(fileContent, sheetName)`: Importa dados de um conteúdo CSV para uma aba.
// - `importAlunosFromSheet(sourceSpreadsheetId, sourceSheetName)`: Importa alunos de outra planilha.
// - `processImportedData(data, targetSheetName)`: Processa e insere os dados importados na planilha.
//
// Observações: A validação robusta dos dados é crucial durante a importação para evitar
//              corrupção ou inconsistência dos dados existentes.

/**
 * Importa dados de um conteúdo CSV para uma aba da planilha.
 * @param {string} fileContent - Conteúdo do arquivo CSV
 * @param {string} sheetName - Nome da aba de destino
 * @param {Object} [options] - Opções de importação (delimiter, skipHeader, validate, etc.)
 * @return {Object} Resultado da importação com estatísticas
 */
function importFromCsv(fileContent, sheetName, options) {
  try {
    if (!fileContent) {
      throw new Error('Conteúdo do arquivo não pode ser vazio');
    }

    if (!sheetName) {
      throw new Error('Nome da aba de destino é obrigatório');
    }

    options = options || {};
    var delimiter = options.delimiter || ',';
    var skipHeader = options.skipHeader !== false; // default true
    var validate = options.validate !== false; // default true

    // Parse do CSV
    var lines = fileContent.split(/\r?\n/);
    var headers = [];
    var dataRows = [];

    if (lines.length === 0) {
      throw new Error('Arquivo CSV vazio');
    }

    // Processa cabeçalho
    if (skipHeader && lines.length > 0) {
      headers = parseCsvLine_(lines[0], delimiter);
      dataRows = lines.slice(1);
    } else {
      dataRows = lines;
    }

    // Parse das linhas de dados
    var parsedData = [];
    var errors = [];

    for (var i = 0; i < dataRows.length; i++) {
      if (!dataRows[i].trim()) continue; // Pula linhas vazias

      try {
        var values = parseCsvLine_(dataRows[i], delimiter);
        
        // Cria objeto se houver cabeçalhos
        if (headers.length > 0) {
          var obj = {};
          for (var j = 0; j < headers.length; j++) {
            obj[headers[j]] = values[j] || '';
          }
          parsedData.push(obj);
        } else {
          parsedData.push(values);
        }
      } catch (parseError) {
        errors.push({ line: i + 1, error: parseError.message });
      }
    }

    // Validação dos dados
    if (validate && typeof validateImportedData_ === 'function') {
      var validationResult = validateImportedData_(parsedData, sheetName);
      if (!validationResult.success) {
        errors = errors.concat(validationResult.errors || []);
      }
    }

    // Processa e insere os dados
    var result = processImportedData(parsedData, sheetName);

    return {
      success: result.success,
      imported: result.imported || 0,
      skipped: parsedData.length - (result.imported || 0),
      total: parsedData.length,
      errors: errors,
      timestamp: new Date().toISOString()
    };
  } catch (error) {
    Logger.log("Erro em importFromCsv: " + error.message);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Importa alunos de outra planilha do Google Sheets.
 * @param {string} sourceSpreadsheetId - ID da planilha de origem
 * @param {string} [sourceSheetName] - Nome da aba de origem (padrão: 'Alunos')
 * @param {Object} [options] - Opções de importação
 * @return {Object} Resultado da importação
 */
function importAlunosFromSheet(sourceSpreadsheetId, sourceSheetName, options) {
  try {
    if (!sourceSpreadsheetId) {
      throw new Error('ID da planilha de origem é obrigatório');
    }

    sourceSheetName = sourceSheetName || 'Alunos';
    options = options || {};

    // Abre planilha de origem
    var sourceSpreadsheet;
    try {
      sourceSpreadsheet = SpreadsheetApp.openById(sourceSpreadsheetId);
    } catch (e) {
      throw new Error('Não foi possível abrir a planilha de origem: ' + e.message);
    }

    var sourceSheet = sourceSpreadsheet.getSheetByName(sourceSheetName);
    if (!sourceSheet) {
      throw new Error('Aba não encontrada na planilha de origem: ' + sourceSheetName);
    }

    // Lê dados da aba de origem
    var sourceData = sourceSheet.getDataRange().getValues();
    if (sourceData.length < 2) {
      throw new Error('Planilha de origem está vazia ou sem dados');
    }

    // Primeira linha são os cabeçalhos
    var headers = sourceData[0].map(function(h) { return String(h || '').trim(); });
    var dataRows = sourceData.slice(1);

    // Converte para objetos
    var alunos = dataRows.map(function(row) {
      var obj = {};
      headers.forEach(function(header, index) {
        obj[header] = row[index];
      });
      return obj;
    }).filter(function(aluno) {
      // Filtra linhas vazias
      return Object.keys(aluno).some(function(key) {
        return aluno[key] !== '' && aluno[key] !== null;
      });
    });

    // Processa e insere os dados
    var result = processImportedData(alunos, 'Alunos');

    // Registra auditoria
    try {
      if (typeof logAudit === 'function') {
        logAudit(
          'system',
          'IMPORT',
          'Alunos',
          sourceSpreadsheetId,
          {
            source: sourceSpreadsheetId,
            imported: result.imported,
            total: alunos.length
          }
        );
      }
    } catch (auditError) {
      Logger.log("Aviso: não foi possível registrar auditoria: " + auditError.message);
    }

    return {
      success: result.success,
      imported: result.imported || 0,
      total: alunos.length,
      timestamp: new Date().toISOString()
    };
  } catch (error) {
    Logger.log("Erro em importAlunosFromSheet: " + error.message);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Processa e insere dados importados na planilha de destino.
 * @param {Array<Object>} data - Array de objetos com dados a importar
 * @param {string} targetSheetName - Nome da aba de destino
 * @param {Object} [options] - Opções de processamento (skipDuplicates, updateExisting, etc.)
 * @return {Object} Resultado do processamento
 */
function processImportedData(data, targetSheetName, options) {
  try {
    if (!data || !Array.isArray(data) || data.length === 0) {
      return {
        success: true,
        imported: 0,
        message: 'Nenhum dado para importar'
      };
    }

    if (!targetSheetName) {
      throw new Error('Nome da aba de destino é obrigatório');
    }

    options = options || {};
    var skipDuplicates = options.skipDuplicates !== false; // default true
    var updateExisting = options.updateExisting || false;

    var imported = 0;
    var skipped = 0;
    var updated = 0;
    var errors = [];

    // Determina a função de criação apropriada
    var createFunction = null;
    var updateFunction = null;
    var findFunction = null;

    switch (targetSheetName) {
      case 'Alunos':
        createFunction = typeof createAluno === 'function' ? createAluno : null;
        updateFunction = typeof updateAluno === 'function' ? updateAluno : null;
        findFunction = typeof getAlunoById === 'function' ? getAlunoById : null;
        break;
      
      case 'Usuarios':
      case 'Users':
        createFunction = typeof createUser === 'function' ? createUser : null;
        updateFunction = typeof updateUser === 'function' ? updateUser : null;
        findFunction = typeof getUserById === 'function' ? getUserById : null;
        break;
      
      case 'Simulacoes':
        createFunction = typeof startSimulation === 'function' ? startSimulation : null;
        break;
      
      case 'Pontuacoes':
        createFunction = typeof recordPontuacao === 'function' ? recordPontuacao : null;
        break;
    }

    // Processa cada registro
    for (var i = 0; i < data.length; i++) {
      var record = data[i];
      
      try {
        // Verifica duplicatas se solicitado
        var isDuplicate = false;
        if (skipDuplicates && record.ID && findFunction) {
          try {
            var existing = findFunction(record.ID);
            if (existing && existing.success) {
              isDuplicate = true;
              
              // Atualiza se solicitado
              if (updateExisting && updateFunction) {
                var updateResult = updateFunction(record.ID, record);
                if (updateResult.success) {
                  updated++;
                } else {
                  errors.push({ record: i + 1, error: 'Falha ao atualizar: ' + (updateResult.message || 'erro desconhecido') });
                  skipped++;
                }
              } else {
                skipped++;
              }
            }
          } catch (e) {
            // Se não encontrar, não é duplicata
          }
        }

        // Insere novo registro se não for duplicata
        if (!isDuplicate) {
          if (createFunction) {
            var result = createFunction(record);
            if (result.success) {
              imported++;
            } else {
              errors.push({ record: i + 1, error: result.message || 'Falha ao criar registro' });
              skipped++;
            }
          } else {
            // Fallback: usa wtgCreateRecord_ genérico
            if (typeof wtgCreateRecord_ === 'function') {
              var headers = Object.keys(record);
              var createResult = wtgCreateRecord_(targetSheetName, headers, record, {});
              if (createResult.success) {
                imported++;
              } else {
                errors.push({ record: i + 1, error: createResult.message || 'Falha ao criar registro' });
                skipped++;
              }
            } else {
              throw new Error('Função de criação não disponível para ' + targetSheetName);
            }
          }
        }
      } catch (recordError) {
        errors.push({ record: i + 1, error: recordError.message });
        skipped++;
      }
    }

    return {
      success: errors.length === 0 || imported > 0,
      imported: imported,
      updated: updated,
      skipped: skipped,
      errors: errors,
      timestamp: new Date().toISOString()
    };
  } catch (error) {
    Logger.log("Erro em processImportedData: " + error.message);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Funções auxiliares privadas
 */

function parseCsvLine_(line, delimiter) {
  delimiter = delimiter || ',';
  var values = [];
  var currentValue = '';
  var inQuotes = false;

  for (var i = 0; i < line.length; i++) {
    var char = line[i];

    if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === delimiter && !inQuotes) {
      values.push(currentValue.trim());
      currentValue = '';
    } else {
      currentValue += char;
    }
  }

  // Adiciona o último valor
  values.push(currentValue.trim());

  return values;
}

function validateImportedData_(data, sheetName) {
  var errors = [];

  // Validações específicas por tipo de aba
  if (sheetName === 'Alunos') {
    data.forEach(function(record, index) {
      if (!record.Nome && !record.nome) {
        errors.push({ record: index + 1, error: 'Nome é obrigatório' });
      }
    });
  } else if (sheetName === 'Usuarios' || sheetName === 'Users') {
    data.forEach(function(record, index) {
      if (!record.Username && !record.username) {
        errors.push({ record: index + 1, error: 'Username é obrigatório' });
      }
      if (!record.Password && !record.password) {
        errors.push({ record: index + 1, error: 'Password é obrigatório' });
      }
    });
  }

  return {
    success: errors.length === 0,
    errors: errors
  };
}
