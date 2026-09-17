// PsychomotricityEval.gs
//
// Funcionalidade Principal: Funções específicas para avaliação da psicomotricidade dos alunos.
//
// Descrição: Este script contém a lógica para avaliar e registrar o desempenho dos alunos
//            em aspectos psicomotores durante as simulações de trânsito. Isso inclui
//            esquema corporal, lateralidade, coordenação motora global, organização espacial
//            e temporal, conforme detalhado no artigo.
//
// Integrações:
// - Google Planilha (aba `Aval_Psicomotricidade`): Armazenamento dos resultados.
// - PedagogicalEval.gs (helpers wtgEval*_): cálculo e persistência por dimensão.
// - SimulacaoService.gs: associa a avaliação a uma simulação específica.
//
// Funções Principais:
// - `evaluatePsychomotricity(simulacaoId, alunoId, data)`: Avalia e registra os aspectos psicomotores.
// - `getPsychomotricityScores(alunoId)`: Retorna as pontuações psicomotoras de um aluno.
// - `analyzePsychomotricityTrends(alunoId)`: Analisa tendências de desempenho psicomotor.
//
// Observações: Indicadores sugeridos: esquema_corporal, lateralidade, coordenacao_global,
//              organizacao_espacial, organizacao_temporal (0-100).

var DIM_PSICOMOTRICIDADE = 'Psicomotricidade';

function evaluatePsychomotricity(simulacaoId, alunoId, data) {
  return wtgEvalRecord_(DIM_PSICOMOTRICIDADE, simulacaoId, alunoId, data);
}

function getPsychomotricityScores(alunoId) {
  return wtgEvalScores_(DIM_PSICOMOTRICIDADE, alunoId);
}

function analyzePsychomotricityTrends(alunoId) {
  return wtgEvalTrends_(DIM_PSICOMOTRICIDADE, alunoId);
}
