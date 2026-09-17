// ExecutiveFuncEval.gs
//
// Funcionalidade Principal: Funções específicas para avaliação das funções executivas dos alunos.
//
// Descrição: Este script contém a lógica para avaliar e registrar o desempenho dos alunos
//            em relação às funções executivas, como controle inibitório, atenção dividida,
//            memória de trabalho e flexibilidade cognitiva, conforme aplicado ao contexto
//            do trânsito.
//
// Integrações:
// - Google Planilha (aba `Aval_FuncoesExecutivas`): Armazenamento dos resultados.
// - PedagogicalEval.gs (helpers wtgEval*_): cálculo e persistência por dimensão.
// - SimulacaoService.gs: associa a avaliação a uma simulação específica.
//
// Funções Principais:
// - `evaluateExecutiveFunctions(simulacaoId, alunoId, data)`: Avalia e registra os aspectos das funções executivas.
// - `getExecutiveFunctionScores(alunoId)`: Retorna as pontuações das funções executivas de um aluno.
// - `analyzeExecutiveFunctionTrends(alunoId)`: Analisa tendências de desempenho das funções executivas.
//
// Observações: Indicadores sugeridos: controle_inibitorio, atencao_dividida, memoria_trabalho,
//              flexibilidade_cognitiva (0-100).

var DIM_FUNCOES_EXECUTIVAS = 'FuncoesExecutivas';

function evaluateExecutiveFunctions(simulacaoId, alunoId, data) {
  return wtgEvalRecord_(DIM_FUNCOES_EXECUTIVAS, simulacaoId, alunoId, data);
}

function getExecutiveFunctionScores(alunoId) {
  return wtgEvalScores_(DIM_FUNCOES_EXECUTIVAS, alunoId);
}

function analyzeExecutiveFunctionTrends(alunoId) {
  return wtgEvalTrends_(DIM_FUNCOES_EXECUTIVAS, alunoId);
}
