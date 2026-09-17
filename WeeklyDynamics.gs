// WeeklyDynamics.gs
//
// Funcionalidade Principal: Gerencia a lógica e o fluxo das dinâmicas semanais do projeto.
//
// Descrição: Orquestra as atividades e objetivos pedagógicos das quatro semanas temáticas do
//            projeto "Brincando no Trânsito Seguro". Hospeda o plano semanal e os helpers
//            compartilhados usados pelos módulos Semana1-4Logic.
//
// Integrações:
// - ConfigService.gs (aba Settings): semana atual (`current_week`).
// - Semana1-4Logic.gs: lógica específica de cada semana.
// - EventLogger.gs / UserService.gs (wtg*): registro e persistência.
//
// Funções Principais:
// - `getCurrentWeek()`: Retorna a semana atual do projeto (1-4).
// - `setupWeek(weekNumber)`: Define e inicializa a semana atual.
// - `getWeeklyContent(weekNumber)`: Retorna o conteúdo pedagógico (tema, plano, atividades).
// - `advanceToNextWeek()`: Avança o projeto para a próxima semana (até 4).

var WEEKLY_PLAN = {
  1: { tema: 'Semiótica, Percepção Visual e Leitura da Via', atividades: [
    { id: 's1_pintura', nome: 'Pintura do pátio (faixas, placas e sentidos de circulação)', dimensao: 'Psicomotricidade' },
    { id: 's1_debate', nome: 'Debate: que regra aparece em cada sinal?', dimensao: 'Cognitiva' },
    { id: 's1_jogo', nome: 'Jogo de controle inibitório (pare/siga)', dimensao: 'FuncoesExecutivas' }
  ]},
  2: { tema: 'Coordenação, Pedestre e Controle de Velocidade', atividades: [
    { id: 's2_tremzinho', nome: 'Veículos-tremzinho (coordenação coletiva)', dimensao: 'Psicomotricidade' },
    { id: 's2_reduza', nome: 'Placa "Reduza a Velocidade" em via interna de superquadra', dimensao: 'FuncoesExecutivas' },
    { id: 's2_policia', nome: 'Introdução do papel da polícia e dos agentes de trânsito', dimensao: 'MarcoLegal' }
  ]},
  3: { tema: 'Prioridades, Sinalização e Tomada de Decisão', atividades: [
    { id: 's3_cruzamento', nome: 'Cruzamentos, retornos e prioridade', dimensao: 'Cognitiva' },
    { id: 's3_preferencia', nome: 'Placa "Dê a Preferência" na entrada da quadra vizinha', dimensao: 'MarcoLegal' },
    { id: 's3_atencao', nome: 'Atenção dividida e decisão rápida', dimensao: 'FuncoesExecutivas' }
  ]},
  4: { tema: 'Consolidação Ética, Regras e Avaliação', atividades: [
    { id: 's4_cartazes', nome: 'Criação de cartazes: regra, lugar e cuidado', dimensao: 'Pedagogica' },
    { id: 's4_deveres', nome: 'Debate: tamanho e idade definem deveres', dimensao: 'MarcoLegal' },
    { id: 's4_guardioes', nome: 'Preparação como guardiões do trânsito', dimensao: 'Pedagogica' }
  ]}
};

function wtgWeekPlan_(week) { return WEEKLY_PLAN[week] || null; }

/**
 * Atividades da semana já incluindo a trilha de cidadania ancorada nas escalas de
 * Brasília (superquadra/escala gregária), quando o módulo de contexto estiver presente.
 */
function wtgWeekActivities_(week) {
  try {
    var p = wtgWeekPlan_(week);
    var base = p ? p.atividades.slice() : [];
    if (typeof getCitizenshipActivities === 'function') {
      try { base = base.concat(getCitizenshipActivities(week)); } catch (e) {}
    }
    return base;
  } catch (error) {
    Logger.log("Erro em wtgWeekActivities_: " + error.message);
    throw error;
  }
}

function wtgInitWeek_(week) {
  var plan = wtgWeekPlan_(week);
  if (!plan) return { success: false, message: 'Semana inválida: ' + week };
  if (typeof logEvent === 'function') { try { logEvent('SEMANA_INICIADA', { week: week, tema: plan.tema }); } catch (e) {} }
  return { success: true, data: { week: week, tema: plan.tema, atividades: plan.atividades } };
}

var ATIVIDADES_SEMANA_SHEET = 'AtividadesSemana';
var ATIVIDADES_SEMANA_HEADERS = ['ID', 'Semana', 'AtividadeID', 'AlunoID', 'Resultado', 'Score', 'CriadoEm', 'AtualizadoEm'];

function wtgEvaluateWeekActivity_(week, activityId, alunoId, result) {
  if (String(alunoId || '').trim() === '') return { success: false, message: 'alunoId obrigatorio.' };
  var atividade = wtgWeekActivities_(week).filter(function (a) { return a.id === activityId; })[0];
  if (!atividade) return { success: false, message: 'Atividade inexistente: ' + activityId };
  var score = (result && result.score !== undefined) ? Number(result.score) : (result === true ? 100 : 0);
  var reg = wtgCreateRecord_(ATIVIDADES_SEMANA_SHEET, ATIVIDADES_SEMANA_HEADERS, {
    Semana: week, AtividadeID: activityId, AlunoID: alunoId,
    Resultado: typeof result === 'object' ? JSON.stringify(result) : String(result),
    Score: score
  }, { required: ['AlunoID'] });

  // Gamificação: converte o desempenho em pontos. Atividades de Cidadania (escala
  // gregária de Brasília) recebem bônus, reforçando o objetivo central do projeto.
  var gamificacao = null;
  if (score > 0 && typeof awardPoints === 'function') {
    var pontos = (atividade.dimensao === 'Cidadania') ? Math.round(score * 1.5) : score;
    try {
      gamificacao = awardPoints(alunoId, pontos,
        'Semana ' + week + ' — ' + atividade.nome + ' (' + atividade.dimensao + ')').data;
    } catch (e) {}
  }

  return { success: true, data: { week: week, atividade: atividade.nome, dimensao: atividade.dimensao, score: score, registro: reg.data, gamificacao: gamificacao } };
}

function getCurrentWeek() {
  var w = (typeof getSetting === 'function') ? getSetting('current_week') : null;
  var n = Number(w);
  return (!isNaN(n) && n >= 1 && n <= 4) ? n : 1;
}

function setupWeek(weekNumber) {
  if (!wtgWeekPlan_(weekNumber)) return { success: false, message: 'Semana inválida: ' + weekNumber };
  if (typeof setSetting === 'function') setSetting('current_week', weekNumber, 'Semana atual do projeto');
  return wtgInitWeek_(weekNumber);
}

function getWeeklyContent(weekNumber) {
  var plan = wtgWeekPlan_(weekNumber);
  if (!plan) return null;
  return {
    week: weekNumber,
    tema: plan.tema,
    atividades: wtgWeekActivities_(weekNumber),
    plano: (typeof getLessonPlan === 'function') ? getLessonPlan(weekNumber) : null,
    situacoesGamificadas: (typeof getTrafficGameSituationsForWeek === 'function') ? getTrafficGameSituationsForWeek(weekNumber) : []
  };
}

function advanceToNextWeek() {
  var atual = getCurrentWeek();
  if (atual >= 4) return { success: false, message: 'Projeto já está na última semana.', week: 4 };
  return setupWeek(atual + 1);
}
