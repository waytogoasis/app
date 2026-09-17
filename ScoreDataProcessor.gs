// ScoreDataProcessor.gs
//
// Funcionalidade Principal: Processa e normaliza dados brutos de pontuações.
//
// Descrição: Limpa, valida e transforma os dados de entrada relacionados às pontuações dos alunos,
//            garantindo formato consistente antes do armazenamento.
//
// Integrações:
// - PontuacaoService.gs: Utiliza para persistir os dados processados.
// - ValidationUtils.gs: Para validar a integridade dos dados.
//
// Funções Principais:
// - `processNewScoreData(rawData)`: Limpa e valida dados de uma nova pontuação (indicadores 0-100).
// - `normalizeScoreValue(score)`: Normaliza o valor da pontuação (número, 0-100, 2 casas).
// - `validateScoreRange(score, min, max)`: Valida se a pontuação está dentro de um intervalo.

function normalizeScoreValue(score) {
  try {
    var n = Number(score);
    if (isNaN(n)) return 0;
    n = Math.max(0, Math.min(100, n));
    return Math.round(n * 100) / 100;
  } catch (error) {
    Logger.log("Erro em normalizeScoreValue: " + error.message);
    throw error;
  }
}

function validateScoreRange(score, min, max) {
  try {
    var n = Number(score);
    if (isNaN(n)) return false;
    var lo = (min === undefined || min === null) ? 0 : Number(min);
    var hi = (max === undefined || max === null) ? 100 : Number(max);
    return n >= lo && n <= hi;
  } catch (error) {
    Logger.log("Erro em validateScoreRange: " + error.message);
    throw error;
  }
}

function processNewScoreData(rawData) {
  try {
    rawData = rawData || {};
    var errors = [];
    if (typeof isNotNullOrEmpty === 'function' ? !isNotNullOrEmpty(rawData.alunoId || rawData.AlunoID) : !(rawData.alunoId || rawData.AlunoID)) {
      errors.push('alunoId obrigatorio.');
    }
    var indicadoresRaw = rawData.indicadores || rawData.pontuacoes || rawData.Pontuacoes || {};
    var indicadores = {};
    Object.keys(indicadoresRaw).forEach(function (k) {
      if (!validateScoreRange(indicadoresRaw[k], 0, 100)) errors.push('Indicador fora do intervalo (0-100): ' + k);
      indicadores[k] = normalizeScoreValue(indicadoresRaw[k]);
    });
    if (errors.length) return { success: false, errors: errors };
    return {
      success: true,
      data: {
        AlunoID: rawData.alunoId || rawData.AlunoID,
        SimulacaoID: rawData.simulacaoId || rawData.SimulacaoID || '',
        indicadores: indicadores
      }
    };
  } catch (error) {
    Logger.log("Erro em processNewScoreData: " + error.message);
    throw error;
  }
}
