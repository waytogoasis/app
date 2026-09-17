// LegalFrameworkEval.gs
//
// Funcionalidade Principal: Funções específicas para avaliação da compreensão do marco legal (CTB/ECA) pelos alunos.
//
// Descrição: Este script contém a lógica para avaliar e registrar o desempenho dos alunos
//            em relação à compreensão e aplicação dos princípios do Código de Trânsito Brasileiro (CTB)
//            e do Estatuto da Criança e do Adolescente (ECA) no contexto das simulações de trânsito.
//            Foca na internalização de regras como "o maior cuida do menor" e a prioridade do pedestre.
//
// Integrações:
// - Google Planilha (aba `Aval_MarcoLegal`): Armazenamento dos resultados.
// - PedagogicalEval.gs (helpers wtgEval*_): cálculo e persistência por dimensão.
// - SimulacaoService.gs: associa a avaliação a uma simulação específica.
//
// Funções Principais:
// - `evaluateLegalFramework(simulacaoId, alunoId, data)`: Avalia e registra a compreensão do marco legal.
// - `getLegalFrameworkScores(alunoId)`: Retorna as pontuações de compreensão do marco legal de um aluno.
// - `analyzeLegalFrameworkTrends(alunoId)`: Analisa tendências de compreensão do marco legal.
//
// Observações: Indicadores sugeridos: prioridade_pedestre, respeito_sinalizacao,
//              maior_cuida_menor, justificativa_legal, reconheceu_sinalizacao,
//              explicou_regra_no_contexto_brasilia, usou_faixa_quando_existente,
//              reduziu_velocidade_via_interna (0-100).

var DIM_MARCO_LEGAL = 'MarcoLegal';

var LEGAL_FRAMEWORK_BRASILIA_INDICATORS = [
  {
    id: 'prioridade_pedestre',
    criterio: 'Reconhece a prioridade do pedestre, especialmente na faixa.',
    evidencia: 'Para antes da faixa, espera a travessia completa e explica por que a preferência protege a vida.'
  },
  {
    id: 'respeito_sinalizacao',
    criterio: 'Reconhece sinalização horizontal, vertical e semafórica.',
    evidencia: 'Nomeia PARE, Dê a Preferência, faixa, semáforo, velocidade e sentido da via antes de agir.'
  },
  {
    id: 'maior_cuida_menor',
    criterio: 'Aplica a regra pedagógica de proteção aos mais vulneráveis.',
    evidencia: 'Cede a vez a criança, idoso, pessoa com deficiência, ciclista ou pedestre em situação de risco.'
  },
  {
    id: 'explicou_regra_no_contexto_brasilia',
    criterio: 'Conecta a regra ao desenho urbano de Brasília.',
    evidencia: 'Explica por que a conduta muda em superquadra, entrequadra, via interna, eixo/eixinho ou quadra vizinha.'
  },
  {
    id: 'reduziu_velocidade_via_interna',
    criterio: 'Compreende que área residencial e escolar pede velocidade compatível.',
    evidencia: 'Reduz ao entrar na via interna e evita tratar a superquadra como atalho.'
  },
  {
    id: 'usou_faixa_quando_existente',
    criterio: 'Escolhe a travessia sinalizada quando ela está disponível.',
    evidencia: 'Procura a faixa próxima, aguarda condição segura e não atravessa no improviso.'
  }
];

function getLegalFrameworkBrasiliaIndicators() {
  try {
    return JSON.parse(JSON.stringify(LEGAL_FRAMEWORK_BRASILIA_INDICATORS));
  } catch (error) {
    Logger.log("Erro em getLegalFrameworkBrasiliaIndicators: " + error.message);
    throw error;
  }
}

function evaluateLegalFramework(simulacaoId, alunoId, data) {
  return wtgEvalRecord_(DIM_MARCO_LEGAL, simulacaoId, alunoId, data);
}

function getLegalFrameworkScores(alunoId) {
  return wtgEvalScores_(DIM_MARCO_LEGAL, alunoId);
}

function analyzeLegalFrameworkTrends(alunoId) {
  return wtgEvalTrends_(DIM_MARCO_LEGAL, alunoId);
}
