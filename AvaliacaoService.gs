// AvaliacaoService.gs
//
// Funcionalidade Principal: Gerencia a aplicação e o gerenciamento de rubricas de avaliação.
//
// Descrição: Este script define e aplica as rubricas de avaliação para as pontuações dos alunos.
//            Ele interage com a aba `AvaliacaoRubricas` da Google Planilha para buscar os critérios
//            e níveis de desempenho, permitindo uma avaliação consistente e padronizada.
//
// Integrações:
// - Google Planilha (aba `AvaliacaoRubricas`): Leitura e gerenciamento das rubricas.
// - UserService.gs (wtg* helpers): camada de CRUD reutilizável.
// - PontuacaoService.gs: Utiliza as rubricas para interpretar e registrar as pontuações.
//
// Funções Principais:
// - `getRubricas()`: Retorna todas as rubricas de avaliação definidas.
// - `applyRubrica(pontuacoes, rubricaId)`: Aplica uma rubrica específica a um conjunto de pontuações.
// - `createRubrica(rubricaData)`: Adiciona uma nova rubrica de avaliação.
// - `updateRubrica(rubricaId, newRubricaData)`: Atualiza uma rubrica existente.
// - `deleteRubrica(rubricaId)`: Remove (logicamente) uma rubrica de avaliação.

var RUBRICAS_SHEET = 'AvaliacaoRubricas';
var RUBRICAS_HEADERS = ['ID', 'Nome', 'Criterios', 'Status', 'CriadoEm', 'AtualizadoEm'];

function getRubricas() {
  try {
    return wtgReadObjects_(RUBRICAS_SHEET)
      .filter(function (r) { return String(r.Status || '').toLowerCase() !== 'inativo'; })
      .map(function (r) {
        try { r.CriteriosParsed = JSON.parse(r.Criterios || '[]'); } catch (e) { r.CriteriosParsed = []; }
        r.assessmentInterpretation = {
          assessmentType: 'formative_contextualized',
          rubricRole: 'organizes_human_judgment',
          humanJudgmentRequired: true,
          claims: {
            exclusiveAssessment: false,
            stressReductionDemonstrated: false
          },
          note: 'A rubrica organiza o julgamento docente; resultados são evidências parciais e contextuais.'
        };
        return r;
      });
  } catch (error) {
    Logger.log("Erro em getRubricas: " + error.message);
    throw error;
  }
}

function createRubrica(rubricaData) {
  rubricaData = rubricaData || {};
  var criterios = rubricaData.criterios || rubricaData.Criterios || [];
  return wtgCreateRecord_(RUBRICAS_SHEET, RUBRICAS_HEADERS, {
    Nome: rubricaData.nome || rubricaData.Nome || '',
    Criterios: typeof criterios === 'string' ? criterios : JSON.stringify(criterios),
    Status: 'ativo'
  }, { required: ['Nome'] });
}

function updateRubrica(rubricaId, newRubricaData) {
  try {
    try {
      newRubricaData = newRubricaData || {};
      var updates = {};
      if (newRubricaData.nome || newRubricaData.Nome) updates.Nome = newRubricaData.nome || newRubricaData.Nome;
      var criterios = newRubricaData.criterios || newRubricaData.Criterios;
      if (criterios !== undefined) updates.Criterios = typeof criterios === 'string' ? criterios : JSON.stringify(criterios);
      return wtgUpdateRecordById_(RUBRICAS_SHEET, rubricaId, updates);
    } catch (error) {
      Logger.log("Erro em updateRubrica: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em updateRubrica: " + error.message);
    throw error;
  }
}

function deleteRubrica(rubricaId) {
  return wtgUpdateRecordById_(RUBRICAS_SHEET, rubricaId, { Status: 'inativo' });
}

// Aplica a rubrica: cada critério { chave, peso }. Retorna uma síntese ponderada
// 0-100 acompanhada do contrato de interpretação humana. A nota não é um
// diagnóstico automático, uma avaliação exclusiva ou uma medida de estresse.
function applyRubrica(pontuacoes, rubricaId) {
  try {
    try {
      var found = wtgFindRecordById_(RUBRICAS_SHEET, rubricaId);
      if (!found.success) return { success: false, message: 'Rubrica nao encontrada.' };
      var criterios;
      try { criterios = JSON.parse(found.data.Criterios || '[]'); } catch (e) { criterios = []; }
      pontuacoes = pontuacoes || {};
      var somaPesos = 0, somaNotas = 0, detalhes = [];
      criterios.forEach(function (c) {
        var chave = c.chave || c.key || c.criterio;
        var peso = Number(c.peso || c.weight || 1);
        var valor = Number(pontuacoes[chave] || 0);
        somaPesos += peso;
        somaNotas += valor * peso;
        detalhes.push({ criterio: chave, valor: valor, peso: peso });
      });
      var notaFinal = somaPesos ? Math.round((somaNotas / somaPesos) * 100) / 100 : 0;
      return {
        success: true,
        data: {
          rubricaId: rubricaId,
          notaFinal: notaFinal,
          detalhes: detalhes,
          assessmentInterpretation: {
            assessmentType: 'formative_contextualized',
            rubricRole: 'organizes_human_judgment',
            humanJudgmentRequired: true,
            claims: {
              exclusiveAssessment: false,
              stressReductionDemonstrated: false
            },
            note: 'A rubrica organiza o julgamento docente; resultados são evidências parciais e contextuais.'
          }
        }
      };
    } catch (error) {
      Logger.log("Erro em applyRubrica: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em applyRubrica: " + error.message);
    throw error;
  }
}
