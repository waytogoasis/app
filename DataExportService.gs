// DataExportService.gs
//
// Funcionalidade Principal: Exporta dados do sistema em diferentes formatos.
//
// Descrição: Exporta dados de abas para CSV e gera payloads de relatório (texto/HTML) para
//            posterior conversão em PDF. A geração de PDF em si depende de DriveApp/HtmlService
//            no ambiente GAS; aqui produzimos o conteúdo de forma determinística e testável.
//
// Integrações:
// - SpreadsheetUtils.gs (getData): fonte dos dados.
// - SimulacaoService.gs: dados de simulações.
//
// Funções Principais:
// - `exportToCsv(sheetName)`: Exporta o conteúdo de uma aba para string CSV.
// - `exportAlunoDataToPdf(alunoId)`: Gera o conteúdo (HTML) do relatório de um aluno para PDF.
// - `exportAllSimulations()`: Exporta todas as simulações em CSV.

function des_csvCell_(value) {
  try {
    var s = String(value === null || value === undefined ? '' : value);
    if (/[",\n]/.test(s)) s = '"' + s.replace(/"/g, '""') + '"';
    return s;
  } catch (error) {
    Logger.log("Erro em des_csvCell_: " + error.message);
    throw error;
  }
}

function des_rowsToCsv_(rows) {
  try {
    return (rows || []).map(function (row) {
      return row.map(des_csvCell_).join(',');
    }).join('\n');
  } catch (error) {
    Logger.log("Erro em des_rowsToCsv_: " + error.message);
    throw error;
  }
}

function exportToCsv(sheetName) {
  var values = (typeof getData === 'function') ? getData(sheetName) : [];
  return des_rowsToCsv_(values);
}

function exportAllSimulations() {
  try {
    var sims = (typeof getAllSimulations === 'function') ? getAllSimulations() : [];
    if (!sims.length) return '';
    var headers = Object.keys(sims[0]).filter(function (k) { return k === k.toUpperCase() || /^[A-Z]/.test(k); });
    // Usa apenas chaves "originais" (com inicial maiúscula) para evitar duplicar as minúsculas.
    if (!headers.length) headers = Object.keys(sims[0]);
    var rows = [headers];
    sims.forEach(function (s) { rows.push(headers.map(function (h) { return s[h]; })); });
    return des_rowsToCsv_(rows);
  } catch (error) {
    Logger.log("Erro em exportAllSimulations: " + error.message);
    throw error;
  }
}

function exportAlunoDataToPdf(alunoId) {
  try {
    var rel = (typeof generateRelatorioAluno === 'function') ? generateRelatorioAluno(alunoId) : null;
    if (rel && rel.success === false) return { success: false, message: 'Aluno não encontrado.' };
    var dados = rel && rel.data ? rel.data : {};
    var nome = dados.aluno ? (dados.aluno.Nome || dados.aluno.nome || alunoId) : alunoId;
    var media = dados.progresso ? dados.progresso.media : 0;
    var html = '<h1>Relatório do Aluno</h1>' +
      '<p><strong>Nome:</strong> ' + nome + '</p>' +
      '<p><strong>Média:</strong> ' + media + '</p>' +
      '<p><strong>Simulações:</strong> ' + ((dados.simulacoes || []).length) + '</p>' +
      '<p>Gerado em ' + new Date().toISOString() + '</p>';
    // Best-effort para PDF real quando DriveApp/Utilities estiverem disponíveis.
    var pdfGerado = false;
    try {
      if (typeof Utilities !== 'undefined' && Utilities.newBlob) {
        Utilities.newBlob(html, 'text/html', 'relatorio_' + alunoId + '.html');
        pdfGerado = true;
      }
    } catch (e) { pdfGerado = false; }
    return { success: true, data: { alunoId: alunoId, formato: 'html', conteudo: html, pdfBlobCriado: pdfGerado } };
  } catch (error) {
    Logger.log("Erro em exportAlunoDataToPdf: " + error.message);
    throw error;
  }
}
