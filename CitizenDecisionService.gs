/**
 * Ciclo curto de decisão cidadã: escolher, justificar, receber consequência e
 * revisar a própria ação. A revisão é obrigatória para concluir a atividade.
 */
var CITIZEN_DECISIONS_SHEET = 'Decisoes_Cidadas';
var CITIZEN_DECISIONS_HEADERS = [
  'ID', 'UsuarioID', 'CenarioID', 'Escolha', 'Justificativa', 'Resultado',
  'Indicadores', 'Reflexao', 'ProximaAcao', 'Status', 'CriadoEm', 'AtualizadoEm'
];

function getCitizenshipDecisionScenario() {
  return {
    id: 'quadra_vizinha_faixa_01',
    title: 'A faixa entre duas superquadras',
    context: 'Você está saindo da quadra 109 para chegar à 108. Um idoso se aproxima da faixa, uma criança brinca perto dos pilotis e um colega sugere cortar caminho pela área verde.',
    question: 'O que você faz antes de seguir?',
    options: [
      { id: 'acelerar', label: 'Acelero para passar antes do idoso chegar à faixa.' },
      { id: 'atalho_verde', label: 'Corto pela área verde para não cruzar com ninguém.' },
      { id: 'parar_observar', label: 'Reduzo, paro, observo a sinalização e cedo a passagem.' }
    ],
    justificationPrompt: 'Explique como sua escolha protege as pessoas e respeita a quadra vizinha.',
    minimumJustificationLength: 20
  };
}

function evaluateCitizenshipDecision(payload) {
  var scenario = getCitizenshipDecisionScenario();
  var choice = String(payload && payload.choice || '').trim();
  var justification = String(payload && payload.justification || '').trim();
  var valid = scenario.options.some(function (option) { return option.id === choice; });
  if (!valid) throw new Error('Escolha uma das três ações do cenário.');
  if (justification.length < scenario.minimumJustificationLength) {
    throw new Error('Explique sua decisão com pelo menos 20 caracteres.');
  }

  var safe = choice === 'parar_observar';
  return {
    scenarioId: scenario.id,
    choice: choice,
    justification: justification.slice(0, 600),
    outcome: safe ? 'segura' : 'precisa_revisao',
    indicators: safe
      ? ['reconheceu_sinalizacao', 'deu_preferencia_pedestre', 'respeitou_area_convivio']
      : [],
    feedback: safe
      ? 'Sua ação cria tempo para ler a sinalização e protege quem está a pé. Agora transforme essa boa decisão em uma regra que você consiga repetir em outro lugar.'
      : 'Essa escolha aumenta o risco ou invade um espaço de convivência. Compare-a com a regra: reduzir, observar a sinalização, ceder a quem está a pé e preservar a área verde.'
  };
}

function submitCitizenshipDecision(userId, payload) {
  var owner = String(userId || '').trim();
  if (!owner) throw new Error('Usuário da atividade não identificado.');
  var evaluation = evaluateCitizenshipDecision(payload || {});
  var created = wtgCreateRecord_(CITIZEN_DECISIONS_SHEET, CITIZEN_DECISIONS_HEADERS, {
    UsuarioID: owner,
    CenarioID: evaluation.scenarioId,
    Escolha: evaluation.choice,
    Justificativa: evaluation.justification,
    Resultado: evaluation.outcome,
    Indicadores: JSON.stringify(evaluation.indicators),
    Reflexao: '',
    ProximaAcao: '',
    Status: 'aguardando_revisao'
  }, { required: ['UsuarioID', 'CenarioID', 'Escolha', 'Justificativa'] });
  if (!created || created.success === false) throw new Error('Não foi possível registrar a decisão.');
  return {
    attemptId: String(created.data.ID || created.data.id || ''),
    outcome: evaluation.outcome,
    indicators: evaluation.indicators,
    feedback: evaluation.feedback,
    requiresReflection: true
  };
}

function completeCitizenshipReflection(userId, payload) {
  var owner = String(userId || '').trim();
  var attemptId = String(payload && payload.attemptId || '').trim();
  var reflection = String(payload && payload.reflection || '').trim();
  var nextAction = String(payload && payload.nextAction || '').trim();
  if (!attemptId) throw new Error('Decisão anterior não identificada.');
  if (reflection.length < 20) throw new Error('Compare sua decisão com a devolutiva em pelo menos 20 caracteres.');
  if (nextAction.length < 15) throw new Error('Escreva uma próxima ação concreta com pelo menos 15 caracteres.');

  var found = wtgFindRecordById_(CITIZEN_DECISIONS_SHEET, attemptId);
  if (!found || !found.success) throw new Error('Decisão não encontrada.');
  var record = found.data || {};
  if (String(record.UsuarioID || record.usuarioid || '') !== owner) {
    throw new Error('Esta decisão pertence a outro usuário.');
  }
  if (String(record.Status || record.status || '') !== 'aguardando_revisao') {
    throw new Error('Esta atividade já foi concluída.');
  }

  var updated = wtgUpdateRecordById_(CITIZEN_DECISIONS_SHEET, attemptId, {
    Reflexao: reflection.slice(0, 800),
    ProximaAcao: nextAction.slice(0, 500),
    Status: 'concluida'
  });
  if (!updated || updated.success === false) throw new Error('Não foi possível concluir a revisão.');
  return { attemptId: attemptId, status: 'concluida', message: 'Ciclo concluído: decisão, consequência e revisão ficaram registradas.' };
}

function getCitizenshipDecisionProgress(userId) {
  var owner = String(userId || '').trim();
  var rows = wtgReadObjects_(CITIZEN_DECISIONS_SHEET).filter(function (row) {
    return String(row.UsuarioID || row.usuarioid || '') === owner;
  });
  return {
    attempts: rows.length,
    completed: rows.filter(function (row) { return String(row.Status || row.status || '') === 'concluida'; }).length,
    awaitingReflection: rows.filter(function (row) { return String(row.Status || row.status || '') === 'aguardando_revisao'; }).length
  };
}
