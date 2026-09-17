// BackupRestoreService.gs
//
// Funcionalidade Principal: Gerencia o backup e restauração dos dados da Google Planilha.
//
// Descrição: Este script fornece funções para criar cópias de segurança da planilha principal
//            e, se necessário, restaurar dados a partir de um backup. É uma funcionalidade
//            crítica para a recuperação de desastres e a integridade dos dados.
//
// Integrações:
// - Google Drive API: Para armazenar os arquivos de backup.
// - SpreadsheetApp (Apps Script): Para copiar e manipular planilhas.
// - GoogleDriveManager.gs: Para interagir com o Google Drive.
//
// Funções Principais:
// - `createFullBackup()`: Cria um backup completo da planilha principal no Google Drive.
// - `restoreFromBackup(backupFileId)`: Restaura a planilha a partir de um arquivo de backup.
// - `listBackups()`: Lista os backups disponíveis no Google Drive.
//
// Observações: A automação de backups é essencial para a segurança dos dados do projeto.

function createFullBackup() {
  try {
    var ss = wtgGetSpreadsheet_();
    if (!ss) return { success: false, message: 'Planilha principal indisponivel.' };
    if (typeof createConfiguredSpreadsheetBackup === 'function') {
      var file = createConfiguredSpreadsheetBackup(ss.getId(), 'way-to-go-full-backup');
      return { success: true, fileId: file.getId(), name: file.getName(), url: file.getUrl() };
    }
    var copy = DriveApp.getFileById(ss.getId()).makeCopy('way-to-go-full-backup-' + new Date().toISOString());
    return { success: true, fileId: copy.getId(), name: copy.getName(), url: copy.getUrl() };
  } catch (error) {
    Logger.log("Erro em createFullBackup: " + error.message);
    throw error;
  }
}

function restoreFromBackup(backupFileId) {
  try {
    if (!backupFileId) return { success: false, message: 'Informe o ID do backup.' };
    var backup = SpreadsheetApp.openById(backupFileId);
    var target = wtgGetSpreadsheet_();
    if (!target) return { success: false, message: 'Planilha principal indisponivel.' };
    var restored = [];
    backup.getSheets().forEach(function (sourceSheet) {
      var targetSheet = target.getSheetByName(sourceSheet.getName()) || target.insertSheet(sourceSheet.getName());
      targetSheet.clearContents();
      var rows = sourceSheet.getLastRow();
      var cols = sourceSheet.getLastColumn();
      if (rows && cols) {
        targetSheet.getRange(1, 1, rows, cols).setValues(sourceSheet.getRange(1, 1, rows, cols).getValues());
      }
      restored.push(sourceSheet.getName());
    });
    return { success: true, restoredSheets: restored };
  } catch (error) {
    Logger.log("Erro em restoreFromBackup: " + error.message);
    throw error; // Re-lança para tratamento superior
  }
}

function listBackups() {
  if (typeof listConfiguredBackups === 'function') {
    return { success: true, items: listConfiguredBackups(50) };
  }
  return { success: true, items: [] };
}
