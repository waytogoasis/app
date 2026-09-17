// PDFGenerator.gs
//
// Funcionalidade Principal: Gera documentos PDF a partir de dados ou templates HTML.
//
// Descrição: Este script é responsável por criar relatórios, certificados ou outros documentos
//            em formato PDF. Pode converter conteúdo HTML em PDF ou gerar PDFs diretamente
//            a partir de dados formatados. Requer a utilização de serviços externos ou
//            bibliotecas para a conversão, pois o Apps Script não possui um gerador de PDF nativo robusto.
//
// Integrações:
// - Google Drive API: Para salvar os PDFs gerados.
// - HtmlService.gs: Para renderizar templates HTML que serão convertidos em PDF.
// - DataExportService.gs: Utiliza este serviço para exportar dados em PDF.
//
// Funções Principais:
// - `generatePdfFromHtml(htmlContent, fileName)`: Converte um conteúdo HTML em um arquivo PDF.
// - `generateReportPdf(reportData, fileName)`: Gera um relatório PDF a partir de dados estruturados.
// - `savePdfToDrive(pdfBlob, folderId, fileName)`: Salva o PDF gerado no Google Drive.
//
// Observações: A geração de PDF no Apps Script pode ser um desafio e pode exigir soluções
//              criativas, como o uso de Google Docs para conversão ou APIs de terceiros.

/**
 * Converte conteúdo HTML em um arquivo PDF.
 * Utiliza Google Docs como intermediário para a conversão.
 * @param {string} htmlContent - Conteúdo HTML a ser convertido
 * @param {string} [fileName] - Nome do arquivo PDF (padrão: 'documento.pdf')
 * @param {Object} [options] - Opções adicionais (orientation, format, etc.)
 * @return {Object} Resultado da operação com blob e URL do PDF
 */
function generatePdfFromHtml(htmlContent, fileName, options) {
  try {
    if (!htmlContent) {
      throw new Error('Conteúdo HTML não pode ser vazio');
    }

    fileName = fileName || 'documento.pdf';
    options = options || {};

    // Remove extensão se fornecida e adiciona .pdf
    var baseName = fileName.replace(/\.(pdf|html)$/i, '');
    var pdfFileName = baseName + '.pdf';
    var tempDocName = baseName + '_temp';

    // Adiciona estilos CSS para melhor renderização
    var styledHtml = 
      '<html><head>' +
      '<style>' +
      'body { font-family: Arial, sans-serif; margin: 20px; font-size: 11pt; }' +
      'h1 { color: #333; font-size: 18pt; margin-bottom: 10px; }' +
      'h2 { color: #555; font-size: 14pt; margin-top: 15px; margin-bottom: 8px; }' +
      'h3 { color: #666; font-size: 12pt; margin-top: 12px; margin-bottom: 6px; }' +
      'table { border-collapse: collapse; width: 100%; margin: 10px 0; }' +
      'th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }' +
      'th { background-color: #f2f2f2; font-weight: bold; }' +
      'p { margin: 8px 0; line-height: 1.4; }' +
      '.header { text-align: center; margin-bottom: 20px; }' +
      '.footer { margin-top: 30px; font-size: 9pt; color: #666; border-top: 1px solid #ddd; padding-top: 10px; }' +
      '</style>' +
      '</head><body>' +
      htmlContent +
      '</body></html>';

    // Cria um documento temporário do Google Docs
    var tempDoc = DocumentApp.create(tempDocName);
    var tempDocId = tempDoc.getId();
    
    try {
      // Insere o conteúdo HTML no documento
      var body = tempDoc.getBody();
      body.clear();
      
      // Apps Script não suporta inserção direta de HTML complexo
      // Workaround: usa appendParagraph para texto simples
      // Para HTML rico, converte para texto com formatação básica
      var textContent = htmlContent
        .replace(/<br\s*\/?>/gi, '\n')
        .replace(/<\/p>/gi, '\n')
        .replace(/<[^>]+>/g, ''); // Remove tags HTML
      
      body.appendParagraph(textContent);

      // Converte o documento para PDF
      var pdfBlob = DriveApp.getFileById(tempDocId)
        .getAs('application/pdf')
        .setName(pdfFileName);

      // Salva o PDF no Drive
      var pdfFile = DriveApp.createFile(pdfBlob);
      var pdfUrl = pdfFile.getUrl();
      var pdfId = pdfFile.getId();

      // Remove o documento temporário
      DriveApp.getFileById(tempDocId).setTrashed(true);

      Logger.log("PDF gerado com sucesso: " + pdfFileName);

      return {
        success: true,
        pdfId: pdfId,
        pdfUrl: pdfUrl,
        fileName: pdfFileName,
        blob: pdfBlob
      };
    } catch (conversionError) {
      // Limpa o documento temporário em caso de erro
      try {
        DriveApp.getFileById(tempDocId).setTrashed(true);
      } catch (e) {}
      throw conversionError;
    }
  } catch (error) {
    Logger.log("Erro em generatePdfFromHtml: " + error.message);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Gera um relatório PDF a partir de dados estruturados.
 * @param {Object} reportData - Dados do relatório (título, seções, tabelas, etc.)
 * @param {string} [fileName] - Nome do arquivo PDF
 * @param {Object} [options] - Opções de formatação
 * @return {Object} Resultado da operação com blob e URL do PDF
 */
function generateReportPdf(reportData, fileName, options) {
  try {
    if (!reportData) {
      throw new Error('Dados do relatório não podem ser vazios');
    }

    fileName = fileName || 'relatorio.pdf';
    options = options || {};

    // Constrói HTML do relatório
    var html = buildReportHtml_(reportData, options);

    // Gera PDF a partir do HTML
    return generatePdfFromHtml(html, fileName, options);
  } catch (error) {
    Logger.log("Erro em generateReportPdf: " + error.message);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Salva um blob PDF no Google Drive.
 * @param {Blob} pdfBlob - Blob do PDF a ser salvo
 * @param {string} [folderId] - ID da pasta de destino (opcional)
 * @param {string} [fileName] - Nome do arquivo (opcional)
 * @return {Object} Resultado da operação com ID e URL do arquivo
 */
function savePdfToDrive(pdfBlob, folderId, fileName) {
  try {
    if (!pdfBlob) {
      throw new Error('Blob PDF não pode ser vazio');
    }

    // Define nome do arquivo se não fornecido
    if (fileName) {
      pdfBlob.setName(fileName);
    }

    var pdfFile;

    // Salva em pasta específica ou raiz do Drive
    if (folderId) {
      try {
        var folder = DriveApp.getFolderById(folderId);
        pdfFile = folder.createFile(pdfBlob);
      } catch (folderError) {
        Logger.log("Aviso: pasta não encontrada, salvando na raiz: " + folderError.message);
        pdfFile = DriveApp.createFile(pdfBlob);
      }
    } else {
      pdfFile = DriveApp.createFile(pdfBlob);
    }

    Logger.log("PDF salvo no Drive: " + pdfFile.getName());

    return {
      success: true,
      fileId: pdfFile.getId(),
      fileUrl: pdfFile.getUrl(),
      fileName: pdfFile.getName()
    };
  } catch (error) {
    Logger.log("Erro em savePdfToDrive: " + error.message);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Gera PDF de relatório de aluno específico.
 * @param {string|number} alunoId - ID do aluno
 * @param {Object} [options] - Opções do relatório
 * @return {Object} Resultado da operação
 */
function generateStudentReportPdf(alunoId, options) {
  try {
    if (!alunoId) {
      throw new Error('ID do aluno é obrigatório');
    }

    options = options || {};

    // Obtém dados do aluno
    var aluno = null;
    if (typeof getAlunoById === 'function') {
      var alunoResp = getAlunoById(alunoId);
      if (alunoResp && alunoResp.success) {
        aluno = alunoResp.data;
      }
    }

    if (!aluno) {
      throw new Error('Aluno não encontrado: ' + alunoId);
    }

    // Obtém pontuações do aluno
    var pontuacoes = [];
    if (typeof getPontuacoesByAluno === 'function') {
      pontuacoes = getPontuacoesByAluno(alunoId);
    }

    // Monta dados do relatório
    var reportData = {
      title: 'Relatório de Desempenho - ' + (aluno.Nome || aluno.nome),
      subtitle: 'Sistema Way To Go',
      date: Utilities.formatDate(new Date(), Session.getScriptTimeZone(), 'dd/MM/yyyy'),
      sections: [
        {
          title: 'Informações do Aluno',
          content: [
            'Nome: ' + (aluno.Nome || aluno.nome),
            'ID: ' + (aluno.ID || aluno.id),
            'Turma: ' + (aluno.TurmaID || aluno.turmaId || 'N/A'),
            'Status: ' + (aluno.Status || aluno.status || 'ativo')
          ]
        },
        {
          title: 'Resumo de Desempenho',
          content: [
            'Total de Simulações: ' + pontuacoes.length,
            'Pontuação Média: ' + calculateAverageScore_(pontuacoes)
          ]
        }
      ]
    };

    // Adiciona tabela de pontuações se houver
    if (pontuacoes.length > 0) {
      reportData.sections.push({
        title: 'Histórico de Pontuações',
        table: {
          headers: ['Data', 'Simulação', 'Pontuação'],
          rows: pontuacoes.map(function(p) {
            return [
              formatDate_(p.CriadoEm || p.criadoEm),
              p.SimulacaoID || p.simulacaoId || 'N/A',
              (p.Total || p.total || 0).toFixed(2)
            ];
          })
        }
      });
    }

    var fileName = 'Relatorio_' + sanitizeFileName_(aluno.Nome || aluno.nome) + '_' + 
                   Utilities.formatDate(new Date(), Session.getScriptTimeZone(), 'yyyyMMdd') + '.pdf';

    return generateReportPdf(reportData, fileName, options);
  } catch (error) {
    Logger.log("Erro em generateStudentReportPdf: " + error.message);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Funções auxiliares privadas
 */

function buildReportHtml_(reportData, options) {
  var html = '<div class="header">';
  html += '<h1>' + (reportData.title || 'Relatório') + '</h1>';
  
  if (reportData.subtitle) {
    html += '<h3>' + reportData.subtitle + '</h3>';
  }
  
  if (reportData.date) {
    html += '<p><em>Data: ' + reportData.date + '</em></p>';
  }
  
  html += '</div>';

  // Adiciona seções
  if (reportData.sections && reportData.sections.length > 0) {
    reportData.sections.forEach(function(section) {
      html += '<h2>' + (section.title || 'Seção') + '</h2>';
      
      // Conteúdo textual
      if (section.content && Array.isArray(section.content)) {
        section.content.forEach(function(line) {
          html += '<p>' + line + '</p>';
        });
      } else if (section.content) {
        html += '<p>' + section.content + '</p>';
      }
      
      // Tabela
      if (section.table) {
        html += buildTableHtml_(section.table);
      }
    });
  }

  // Rodapé
  html += '<div class="footer">';
  html += '<p>Gerado automaticamente pelo Sistema Way To Go em ' + 
          Utilities.formatDate(new Date(), Session.getScriptTimeZone(), 'dd/MM/yyyy HH:mm') + '</p>';
  html += '</div>';

  return html;
}

function buildTableHtml_(tableData) {
  var html = '<table>';
  
  // Cabeçalho
  if (tableData.headers && tableData.headers.length > 0) {
    html += '<tr>';
    tableData.headers.forEach(function(header) {
      html += '<th>' + header + '</th>';
    });
    html += '</tr>';
  }
  
  // Linhas
  if (tableData.rows && tableData.rows.length > 0) {
    tableData.rows.forEach(function(row) {
      html += '<tr>';
      row.forEach(function(cell) {
        html += '<td>' + cell + '</td>';
      });
      html += '</tr>';
    });
  }
  
  html += '</table>';
  return html;
}

function calculateAverageScore_(pontuacoes) {
  if (!pontuacoes || pontuacoes.length === 0) {
    return '0.00';
  }
  
  var soma = pontuacoes.reduce(function(sum, p) {
    return sum + (Number(p.Total || p.total) || 0);
  }, 0);
  
  return (soma / pontuacoes.length).toFixed(2);
}

function formatDate_(dateString) {
  try {
    var date = new Date(dateString);
    return Utilities.formatDate(date, Session.getScriptTimeZone(), 'dd/MM/yyyy');
  } catch (e) {
    return 'N/A';
  }
}

function sanitizeFileName_(name) {
  return String(name || 'arquivo')
    .replace(/[^a-zA-Z0-9_\-]/g, '_')
    .substring(0, 50);
}
