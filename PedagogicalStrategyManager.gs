// PedagogicalStrategyManager.gs
//
// Funcionalidade Principal: Gerencia as estratégias pedagógicas e metodologias de ensino.
//
// Descrição: Define e organiza abordagens pedagógicas (Pedagogia de Projetos, Aprender Brincando,
//            Aprendizagem Significativa) e permite associá-las a atividades.
//
// Integrações:
// - Google Planilha (aba `EstrategiaAtividade`): associações estratégia↔atividade.
// - UserService.gs (wtg* helpers): camada de CRUD reutilizável.
//
// Funções Principais:
// - `getStrategyDescription(strategyId)`: Retorna a descrição de uma estratégia pedagógica.
// - `listAllStrategies()`: Lista todas as estratégias pedagógicas definidas.
// - `applyStrategyToActivity(activityId, strategyId)`: Associa uma estratégia a uma atividade.

var ESTRATEGIAS = {
  pedagogia_projetos: { nome: 'Pedagogia de Projetos', descricao: 'Aprendizagem organizada em torno de projetos investigativos com produto final.' },
  aprender_brincando: { nome: 'Aprender Brincando', descricao: 'Uso do lúdico e de simulações para construção do conhecimento.' },
  aprendizagem_significativa: { nome: 'Aprendizagem Significativa', descricao: 'Conexão do novo conteúdo com conhecimentos prévios do aluno.' },
  mao_na_massa: { nome: 'Mão na Massa', descricao: 'Experimentação prática e investigação ativa.' }
};
var ESTRATEGIA_ATIVIDADE_SHEET = 'EstrategiaAtividade';
var ESTRATEGIA_ATIVIDADE_HEADERS = ['ID', 'AtividadeID', 'EstrategiaID', 'CriadoEm', 'AtualizadoEm'];

function getStrategyDescription(strategyId) {
  return ESTRATEGIAS[strategyId] || null;
}

function listAllStrategies() {
  try {
    return Object.keys(ESTRATEGIAS).map(function (id) {
      return { id: id, nome: ESTRATEGIAS[id].nome, descricao: ESTRATEGIAS[id].descricao };
    });
  } catch (error) {
    Logger.log("Erro em listAllStrategies: " + error.message);
    throw error;
  }
}

function applyStrategyToActivity(activityId, strategyId) {
  try {
    if (!ESTRATEGIAS[strategyId]) return { success: false, message: 'Estrategia invalida: ' + strategyId };
    if (String(activityId || '').trim() === '') return { success: false, message: 'activityId obrigatorio.' };
    return wtgCreateRecord_(ESTRATEGIA_ATIVIDADE_SHEET, ESTRATEGIA_ATIVIDADE_HEADERS, {
      AtividadeID: activityId, EstrategiaID: strategyId
    }, { required: ['AtividadeID'] });
  } catch (error) {
    Logger.log("Erro em applyStrategyToActivity: " + error.message);
    throw error;
  }
}
