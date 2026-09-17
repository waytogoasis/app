// CognitiveEval.gs
//
// Funcionalidade Principal: Funções específicas para avaliação do desenvolvimento cognitivo (Piaget) dos alunos.
//
// Descrição: Este script contém a lógica para avaliar e registrar o desempenho dos alunos
//            em relação aos estágios de desenvolvimento cognitivo de Piaget, focando na
//            compreensão de regras, reversibilidade e descentração, conforme aplicado ao contexto
//            do trânsito.
//
// Integrações:
// - Google Planilha (aba `Aval_Cognitiva`): Armazenamento dos resultados.
// - PedagogicalEval.gs (helpers wtgEval*_): cálculo e persistência por dimensão.
// - SimulacaoService.gs: associa a avaliação a uma simulação específica.
//
// Funções Principais:
// - `evaluateCognitiveDevelopment(simulacaoId, alunoId, data)`: Avalia e registra os aspectos cognitivos.
// - `getCognitiveScores(alunoId)`: Retorna as pontuações cognitivas de um aluno.
// - `analyzeCognitiveTrends(alunoId)`: Analisa tendências de desenvolvimento cognitivo.
//
// Observações: Indicadores sugeridos: compreensao_regras, reversibilidade, descentracao (0-100).

var DIM_COGNITIVA = 'Cognitiva';

function evaluateCognitiveDevelopment(simulacaoId, alunoId, data) {
  return wtgEvalRecord_(DIM_COGNITIVA, simulacaoId, alunoId, data);
}

function getCognitiveScores(alunoId) {
  return wtgEvalScores_(DIM_COGNITIVA, alunoId);
}

function analyzeCognitiveTrends(alunoId) {
  return wtgEvalTrends_(DIM_COGNITIVA, alunoId);
}
