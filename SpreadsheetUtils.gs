// SpreadsheetUtils.gs
//
// Funcionalidade Principal: Fornece funções utilitárias para interação com a Google Planilha.
//
// Descrição: Este script contém funções genéricas para ler, escrever, atualizar e deletar dados
//            em abas específicas de uma Google Planilha. Ele abstrai a complexidade da API do
//            Google Sheets, tornando mais fácil para outros serviços interagirem com os dados.
//            A `SPREADSHEETS_ID` deve ser configurada como propriedade de script (SPREADSHEETS_ID
//            ou SPREADSHEET_ID); na ausência, usa a planilha ativa.
//
// Integrações:
// - Google Planilha: Interage diretamente com a API do Google Sheets.
// - Todos os Services (AuthService, UserService, AlunoService, etc.): Utilizam estas funções para acesso a dados.
//
// Funções Principais:
// - `getSpreadsheetId()`: Retorna o ID da planilha configurada.
// - `getData(sheetName)`: Lê todos os dados de uma aba (matriz incluindo cabeçalho).
// - `getDataAsObjects(sheetName)`: Lê os dados como array de objetos (chave = cabeçalho).
// - `appendRow(sheetName, rowData)`: Adiciona uma nova linha (array ou objeto) a uma aba.
// - `updateRow(sheetName, rowIndex, rowData)`: Atualiza uma linha existente (rowIndex = nº da linha na planilha).
// - `deleteRow(sheetName, rowIndex)`: Deleta uma linha.
// - `findRow(sheetName, columnIndex, searchValue)`: Encontra a 1ª linha com valor em uma coluna (0-based).
//
// Observações: Leituras/escritas em lote (getValues/setValues) para eficiência de quota.

var SPREADSHEETS_ID = ''; // Configure SPREADSHEETS_ID ou SPREADSHEET_ID em Script Properties.

function getSpreadsheetId() {
  try {
    var props = PropertiesService.getScriptProperties();
    var id = props.getProperty('SPREADSHEETS_ID') || props.getProperty('SPREADSHEET_ID');
    if (id) return id;
  } catch (e) {}
  if (SPREADSHEETS_ID) return SPREADSHEETS_ID;
  try {
    var active = getBoundSpreadsheet_();
    if (active) return active.getId();
  } catch (e2) {}
  return null;
}

function getSpreadsheet_() {
  var id = getSpreadsheetId();
  if (id) return SpreadsheetApp.openById(id);
  var active = getBoundSpreadsheet_();
  if (active) return active;
  throw new Error('Nenhuma planilha configurada (defina SPREADSHEETS_ID em Script Properties).');
}

function getSheet_(sheetName) {
  var ss = getSpreadsheet_();
  var sheet = ss.getSheetByName(sheetName);
  if (!sheet && typeof SchemaService !== 'undefined' && SchemaService.ensureSheet) {
    try { SchemaService.ensureSheet(sheetName, { spreadsheet: ss }); sheet = ss.getSheetByName(sheetName); } catch (e) {}
  }
  if (!sheet) throw new Error('Aba não encontrada: ' + sheetName);
  return sheet;
}

function getData(sheetName) {
  try {
    var sheet = getSheet_(sheetName);
    if (sheet.getLastRow() < 1 || sheet.getLastColumn() < 1) return [];
    return sheet.getDataRange().getValues();
  } catch (error) {
    Logger.log("Erro em getData: " + error.message);
    throw error;
  }
}

function getDataAsObjects(sheetName) {
  var values = getData(sheetName);
  if (values.length < 2) return [];
  var headers = values[0].map(function(h) { return String(h || '').trim(); });
  return values.slice(1).map(function(row) {
    var obj = {};
    headers.forEach(function(h, i) { obj[h] = row[i]; });
    return obj;
  });
}

function rowFromObject_(sheet, obj) {
  try {
    try {
      var lastCol = sheet.getLastColumn();
      var headers = sheet.getRange(1, 1, 1, lastCol).getValues()[0].map(function(h) { return String(h || '').trim(); });
      return headers.map(function(h) { return obj[h] !== undefined ? obj[h] : ''; });
    } catch (error) {
      Logger.log("Erro em rowFromObject_: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em rowFromObject_: " + error.message);
    throw error;
  }
}

function appendRow(sheetName, rowData) {
  try {
    var sheet = getSheet_(sheetName);
    var row = Array.isArray(rowData) ? rowData : rowFromObject_(sheet, rowData || {});
    sheet.appendRow(row);
    return sheet.getLastRow();
  } catch (error) {
    Logger.log("Erro em appendRow: " + error.message);
    throw error; // Re-lança para tratamento superior
  }
}

function updateRow(sheetName, rowIndex, rowData) {
  try {
    try {
      var sheet = getSheet_(sheetName);
      var row = Array.isArray(rowData) ? rowData : rowFromObject_(sheet, rowData || {});
      sheet.getRange(rowIndex, 1, 1, row.length).setValues([row]);
      return rowIndex;
    } catch (error) {
      Logger.log("Erro em updateRow: " + error.message);
      throw error; // Re-lança para tratamento superior
    }
  } catch (error) {
    Logger.log("Erro em updateRow: " + error.message);
    throw error;
  }
}

function deleteRow(sheetName, rowIndex) {
  try {
    try {
      var sheet = getSheet_(sheetName);
      sheet.deleteRow(rowIndex);
      return true;
    } catch (error) {
      Logger.log("Erro em deleteRow: " + error.message);
      throw error; // Re-lança para tratamento superior
    }
  } catch (error) {
    Logger.log("Erro em deleteRow: " + error.message);
    throw error;
  }
}

function findRow(sheetName, columnIndex, searchValue) {
  var values = getData(sheetName);
  for (var i = 1; i < values.length; i++) {
    if (String(values[i][columnIndex]) === String(searchValue)) {
      return { rowIndex: i + 1, row: values[i] };
    }
  }
  return null;
}
