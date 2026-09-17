// PontuacaoService.gs
//
// Funcionalidade Principal: Gerencia o registro e a recuperação das pontuações dos alunos.
//
// Descrição: Este script é responsável por armazenar as pontuações detalhadas de cada aluno
//            após as simulações, com base nos indicadores de avaliação (psicomotricidade,
//            funções executivas, etc.). Ele interage com a aba 'Pontuacoes' da Google Planilha.
//
// Integrações:
// - Google Planilha (aba 'Pontuacoes'): Todas as operações de dados são realizadas nesta aba.
// - UserService.gs (wtg* helpers): camada de CRUD reutilizável.
// - SimulacaoService.gs: associa pontuações a simulações específicas.
// - AvaliacaoService.gs: usa rubricas para interpretar as pontuações.
//
// Funções Principais:
// - `recordPontuacao(simulacaoId, alunoId, pontuacoes)`: Registra as pontuações de uma simulação.
// - `getPontuacoesBySimulacao(simulacaoId)`: Retorna todas as pontuações de uma simulação.
// - `getPontuacoesByAluno(alunoId)`: Retorna todas as pontuações de um aluno.
// - `getPontuacaoDetalhada(pontuacaoId)`: Retorna os detalhes (JSON parseado) de uma pontuação.
// - `updatePontuacao(pontuacaoId, newPontuacoes)`: Atualiza as pontuações e recalcula o total.

var PONTUACOES_SHEET = 'Pontuacoes';
var PONTUACOES_HEADERS = ['ID', 'SimulacaoID', 'AlunoID', 'Pontuacoes', 'Total', 'CriadoEm', 'AtualizadoEm'];

function pontuacaoTotal_(pontuacoes) {
  try {
    if (!pontuacoes || typeof pontuacoes !== 'object') return 0;
    var values = Object.keys(pontuacoes).map(function (k) { return Number(pontuacoes[k]) || 0; });
    if (!values.length) return 0;
    return Math.round(values.reduce(function (a, b) { return a + b; }, 0) / values.length * 100) / 100;
  } catch (error) {
    Logger.log("Erro em pontuacaoTotal_: " + error.message);
    throw error;
  }
}

function recordPontuacao(simulacaoId, alunoId, pontuacoes) {
  try {
    if (String(simulacaoId || '').trim() === '' || String(alunoId || '').trim() === '') {
      return { success: false, message: 'simulacaoId e alunoId obrigatorios.' };
    }
    return wtgCreateRecord_(PONTUACOES_SHEET, PONTUACOES_HEADERS, {
      SimulacaoID: simulacaoId,
      AlunoID: alunoId,
      Pontuacoes: JSON.stringify(pontuacoes || {}),
      Total: pontuacaoTotal_(pontuacoes)
    }, { required: ['SimulacaoID', 'AlunoID'] });
  } catch (error) {
    Logger.log("Erro em recordPontuacao: " + error.message);
    throw error;
  }
}

function parsePontuacao_(record) {
  try {
    if (!record) return record;
    var parsed;
    try { parsed = JSON.parse(record.Pontuacoes || '{}'); } catch (e) { parsed = {}; }
    record.PontuacoesParsed = parsed;
    return record;
  } catch (error) {
    Logger.log("Erro em parsePontuacao_: " + error.message);
    throw error;
  }
}

function getPontuacoesBySimulacao(simulacaoId) {
  try {
    return wtgReadObjects_(PONTUACOES_SHEET)
      .filter(function (p) { return String(p.SimulacaoID || p.simulacaoid || '') === String(simulacaoId); })
      .map(parsePontuacao_);
  } catch (error) {
    Logger.log("Erro em getPontuacoesBySimulacao: " + error.message);
    throw error;
  }
}

function getPontuacoesByAluno(alunoId) {
  try {
    return wtgReadObjects_(PONTUACOES_SHEET)
      .filter(function (p) { return String(p.AlunoID || p.alunoid || '') === String(alunoId); })
      .map(parsePontuacao_);
  } catch (error) {
    Logger.log("Erro em getPontuacoesByAluno: " + error.message);
    throw error;
  }
}

function getPontuacaoDetalhada(pontuacaoId) {
  var found = wtgFindRecordById_(PONTUACOES_SHEET, pontuacaoId);
  if (found.success) found.data = parsePontuacao_(found.data);
  return found;
}

function updatePontuacao(pontuacaoId, newPontuacoes) {
  try {
    return wtgUpdateRecordById_(PONTUACOES_SHEET, pontuacaoId, {
      Pontuacoes: JSON.stringify(newPontuacoes || {}),
      Total: pontuacaoTotal_(newPontuacoes)
    });
  } catch (error) {
    Logger.log("Erro em updatePontuacao: " + error.message);
    throw error;
  }
}
