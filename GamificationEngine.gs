// GamificationEngine.gs
//
// Funcionalidade Principal: Implementa a lógica de gamificação para engajar os alunos.
//
// Descrição: Gerencia pontos, níveis, distintivos e ranking para motivar os alunos a
//            participar das simulações e a internalizar comportamentos seguros e cidadãos
//            no trânsito. Os níveis e distintivos são temáticos do projeto "Brincando no
//            Trânsito Seguro" — incluindo o distintivo "Bom Vizinho", ligado ao cenário
//            das superquadras de Brasília (atravessar a quadra vizinha com respeito).
//
// Integrações:
// - Google Planilha (abas `Pontos`, `Rankings`): histórico de pontos e ranking consolidado.
// - UserService.gs (wtg* helpers): camada de CRUD reutilizável.
// - StudentAchievementManager.gs: definição/concessão de distintivos.
// - BrasiliaUrbanScaleContext.gs: cenário de cidadania que dá origem a distintivos.
// - NotificationService.gs (opcional): avisa o aluno sobre pontos, nível e distintivos.
//
// Funções Principais:
// - `awardPoints(alunoId, points, reason)`: Atribui pontos a um aluno (evento idempotente por linha).
// - `awardTrafficSituation(alunoId, situationId, evidence, reason)`: Pontua uma situação
//   concreta de trânsito/orientação em Brasília com bônus por evidências.
// - `getTrafficGameSituations(category)`: Lista situações gamificadas disponíveis.
// - `getAlunoPoints(alunoId)`: Total de pontos acumulados pelo aluno.
// - `checkLevelUp(alunoId)`: Retorna o nível atual e o próximo, a partir do total de pontos.
// - `updateLeaderboard()`: Recalcula e persiste o ranking dos alunos.
// - `getAlunoRank(alunoId)`: Retorna a posição do aluno no ranking.
// - `getGamificationProfile(alunoId)`: Pontos + nível + ranking + distintivos do aluno.
// - `seedTrafficBadges()`: Registra os distintivos padrão (trânsito/cidadania) de forma idempotente.

var PONTOS_SHEET = 'Pontos';
var PONTOS_HEADERS = ['ID', 'AlunoID', 'Pontos', 'Motivo', 'CriadoEm', 'AtualizadoEm'];
var RANKINGS_SHEET = 'Rankings';
var RANKINGS_HEADERS = ['ID', 'AlunoID', 'Total', 'Posicao', 'Nivel', 'AtualizadoEm'];
var GAMIFICATION_EVENTS_SHEET = 'EventosGamificacao';
var GAMIFICATION_EVENTS_HEADERS = [
  'ID', 'AlunoID', 'SituacaoID', 'Categoria', 'PontosBase', 'Bonus', 'Penalidade',
  'Multiplicador', 'Total', 'Evidencias', 'Motivo', 'CriadoEm', 'AtualizadoEm'
];

// Níveis temáticos: do pedestre iniciante ao orientador cidadão da cidade.
var GAMIFICATION_LEVELS = [
  { nivel: 1, nome: 'Aprendiz de Pedestre', min: 0 },
  { nivel: 2, nome: 'Pedestre Atento', min: 100 },
  { nivel: 3, nome: 'Bom Vizinho da Quadra', min: 250 },
  { nivel: 4, nome: 'Guardião da Superquadra', min: 500 },
  { nivel: 5, nome: 'Orientador dos Eixos', min: 800 },
  { nivel: 6, nome: 'Guardião do Trânsito de Brasília', min: 1200 }
];

// Distintivos padrão. `minMedia` é o critério lido por StudentAchievementManager.
var TRAFFIC_BADGES = [
  { id: 'mestre_semaforo', descricao: 'Mestre do Semáforo — domina o pare/siga.', criteria: { minMedia: 60 } },
  { id: 'amigo_pedestre', descricao: 'Amigo do Pedestre — sempre dá a preferência.', criteria: { minMedia: 70 } },
  { id: 'bom_vizinho', descricao: 'Bom Vizinho — atravessa a superquadra vizinha com respeito (escala gregária de Brasília).', criteria: { minMedia: 80 } },
  { id: 'leitor_sinais_brasilia', descricao: 'Leitor dos Sinais de Brasília — reconhece regras no lugar certo: faixa, PARE, preferência e velocidade.', criteria: { minMedia: 82 } },
  { id: 'orientador_eixos', descricao: 'Orientador dos Eixos — usa eixos, eixinhos e entrequadras para explicar rotas seguras.', criteria: { minMedia: 85 } },
  { id: 'guardiao_transito', descricao: 'Guardião do Trânsito — referência de cidadania no trânsito.', criteria: { minMedia: 90 } }
];

// Situações jogáveis: cada item combina trânsito, cidadania e orientação urbana de Brasília.
var TRAFFIC_GAME_SITUATIONS = [
  {
    id: 'faixa_entrequadra_108_109',
    nome: 'Travessia na faixa da entrequadra 108/109',
    categoria: 'pedestre',
    pontosBase: 30,
    contexto: 'Entrequadra comercial compartilhada, com pedestres chegando ao comércio local e às paradas.',
    objetivo: 'Reconhecer a faixa, sinalizar intenção de atravessar e esperar uma passagem segura.',
    pistasUrbanas: ['faixa de pedestre', 'comércio local', 'ponto de ônibus', 'fluxo de entrada da quadra'],
    bonus: [
      { id: 'reconheceu_faixa', pontos: 8 },
      { id: 'fez_contato_visual', pontos: 6 },
      { id: 'aguardou_travessia_segura', pontos: 8 },
      { id: 'explicou_cultura_faixa_df', pontos: 8 }
    ],
    penalidades: [
      { id: 'atravessou_fora_da_faixa', pontos: -10 },
      { id: 'correu_na_travessia', pontos: -6 }
    ]
  },
  {
    id: 'quadra_vizinha_visitante',
    nome: 'Entrando na superquadra vizinha como visitante',
    categoria: 'superquadra',
    pontosBase: 35,
    contexto: 'Acesso à SQN 108 passando pela SQN 109, com via interna, pilotis e área verde.',
    objetivo: 'Chegar à quadra pretendida sem tratar a quadra vizinha como atalho.',
    pistasUrbanas: ['via interna', 'pilotis', 'área verde', 'placa de preferência', 'crianças brincando'],
    bonus: [
      { id: 'reduziu_velocidade_via_interna', pontos: 10 },
      { id: 'deu_preferencia_pedestre', pontos: 8 },
      { id: 'respeitou_area_verde', pontos: 6 },
      { id: 'explicou_regra_visitante', pontos: 8 }
    ],
    penalidades: [
      { id: 'usou_atalho_velocidade', pontos: -12 },
      { id: 'cortou_area_verde', pontos: -10 }
    ]
  },
  {
    id: 'eixinho_tesourinha_preferencia',
    nome: 'Orientação pelo eixinho e tesourinha',
    categoria: 'eixos',
    pontosBase: 40,
    contexto: 'Deslocamento entre quadras usando eixinho, retorno/tesourinha e sinalização de preferência.',
    objetivo: 'Ler sentido de circulação, preferência e retorno antes de avançar.',
    pistasUrbanas: ['eixinho', 'tesourinha', 'retorno', 'Dê a Preferência', 'setas de direção'],
    bonus: [
      { id: 'identificou_eixinho', pontos: 8 },
      { id: 'leu_sentido_da_via', pontos: 8 },
      { id: 'respeitou_preferencia', pontos: 10 },
      { id: 'planejou_retorno_seguro', pontos: 8 }
    ],
    penalidades: [
      { id: 'entrou_contra_mao', pontos: -15 },
      { id: 'ignorou_preferencia', pontos: -12 }
    ]
  },
  {
    id: 'eixo_monumental_fluxo_civico',
    nome: 'Fluxo seguro na escala monumental',
    categoria: 'eixos',
    pontosBase: 35,
    contexto: 'Grandes distâncias, travessias amplas e fluxo intenso no Eixo Monumental.',
    objetivo: 'Planejar travessia e deslocamento com atenção sustentada.',
    pistasUrbanas: ['Eixo Monumental', 'travessia ampla', 'semáforo', 'canteiro central', 'fluxo intenso'],
    bonus: [
      { id: 'planejou_travessia_longa', pontos: 8 },
      { id: 'respeitou_semaforo', pontos: 8 },
      { id: 'manteve_atencao_sustentada', pontos: 8 },
      { id: 'explicou_escala_monumental', pontos: 6 }
    ],
    penalidades: [
      { id: 'atravessou_no_impulso', pontos: -12 },
      { id: 'desconsiderou_fluxo_intenso', pontos: -10 }
    ]
  },
  {
    id: 'rota_asa_norte_sul',
    nome: 'Rota entre Asa Norte e Asa Sul',
    categoria: 'orientacao',
    pontosBase: 40,
    contexto: 'Leitura de endereçamento por asas, quadras, blocos e eixos principais.',
    objetivo: 'Explicar uma rota simples usando referências reais de Brasília.',
    pistasUrbanas: ['Asa Norte', 'Asa Sul', 'SQN/SQS', 'quadra', 'bloco', 'eixo rodoviário'],
    bonus: [
      { id: 'diferenciou_asa_norte_sul', pontos: 8 },
      { id: 'usou_numero_da_quadra', pontos: 8 },
      { id: 'indicou_eixo_principal', pontos: 8 },
      { id: 'escolheu_travessias_seguras', pontos: 10 }
    ],
    penalidades: [
      { id: 'confundiu_sentido_rota', pontos: -8 },
      { id: 'ignorou_travessia_segura', pontos: -12 }
    ]
  },
  {
    id: 'ponto_onibus_comercio_local',
    nome: 'Chegada segura ao ponto de ônibus e comércio local',
    categoria: 'convivencia',
    pontosBase: 30,
    contexto: 'Movimento de pedestres na entrequadra, embarque/desembarque e comércio de vizinhança.',
    objetivo: 'Organizar espera, travessia e convivência sem invadir a via.',
    pistasUrbanas: ['ponto de ônibus', 'calçada', 'comércio local', 'faixa próxima', 'ônibus parado'],
    bonus: [
      { id: 'esperou_na_calcada', pontos: 6 },
      { id: 'nao_passou_na_frente_do_onibus', pontos: 8 },
      { id: 'procurou_faixa_proxima', pontos: 8 },
      { id: 'respeitou_fluxo_pedestres', pontos: 6 }
    ],
    penalidades: [
      { id: 'invadiu_via_espera', pontos: -10 },
      { id: 'atravessou_encoberto_onibus', pontos: -12 }
    ]
  },
  {
    id: 'area_escolar_superquadra',
    nome: 'Entrada e saída da escola na superquadra',
    categoria: 'escola',
    pontosBase: 35,
    contexto: 'Horário de entrada/saída, crianças, famílias, vans, bicicletas e carros em baixa velocidade.',
    objetivo: 'Reconhecer usuários vulneráveis e organizar circulação com calma.',
    pistasUrbanas: ['portão da escola', 'mochilas', 'vans', 'bicicletas', 'faixa', 'via interna'],
    bonus: [
      { id: 'identificou_usuario_vulneravel', pontos: 8 },
      { id: 'manteve_distancia_segura', pontos: 8 },
      { id: 'respeitou_embarque_desembarque', pontos: 8 },
      { id: 'explicou_maior_cuida_menor', pontos: 8 }
    ],
    penalidades: [
      { id: 'pressionou_pedestre', pontos: -10 },
      { id: 'parou_em_local_inseguro', pontos: -8 }
    ]
  }
];

function ge_clone_(value) {
  try {
    return JSON.parse(JSON.stringify(value));
  } catch (error) {
    Logger.log("Erro em ge_clone_: " + error.message);
    throw error;
  }
}

function ge_findTrafficSituation_(situationId) {
  return TRAFFIC_GAME_SITUATIONS.filter(function (s) {
    return String(s.id) === String(situationId);
  })[0] || null;
}

function ge_evidenceValue_(evidence, id) {
  evidence = evidence || {};
  if (evidence[id] !== undefined) return evidence[id];
  return evidence[String(id).toLowerCase()];
}

function ge_indicatorMet_(value) {
  if (value === true) return true;
  if (typeof value === 'number') return value > 0;
  if (typeof value === 'string') {
    var normalized = value.toLowerCase().trim();
    return ['true', 'sim', 'ok', 'feito', 'realizado', '1', '100'].indexOf(normalized) !== -1;
  }
  return false;
}

function getTrafficGameSituations(category) {
  try {
    var list = TRAFFIC_GAME_SITUATIONS;
    if (category !== undefined && category !== null && String(category).trim() !== '') {
      list = list.filter(function (s) { return String(s.categoria) === String(category); });
    }
    return ge_clone_(list);
  } catch (error) {
    Logger.log("Erro em getTrafficGameSituations: " + error.message);
    throw error;
  }
}

function calculateTrafficSituationScore(situationId, evidence) {
  try {
    var situation = ge_findTrafficSituation_(situationId);
    if (!situation) return { success: false, message: 'Situacao gamificada inexistente: ' + situationId };

    evidence = evidence || {};
    var score = Number(evidence.score !== undefined ? evidence.score : evidence.Score);
    var base = situation.pontosBase;
    if (!isNaN(score)) base = Math.round(situation.pontosBase * Math.max(0, Math.min(100, score)) / 100);

    var bonus = 0;
    var bonusAplicados = [];
    (situation.bonus || []).forEach(function (b) {
      var value = ge_evidenceValue_(evidence, b.id);
      if (!ge_indicatorMet_(value)) return;
      bonus += Number(b.pontos) || 0;
      bonusAplicados.push(b.id);
    });

    var penalidade = 0;
    var penalidadesAplicadas = [];
    (situation.penalidades || []).forEach(function (p) {
      var value = ge_evidenceValue_(evidence, p.id);
      if (!ge_indicatorMet_(value)) return;
      penalidade += Number(p.pontos) || 0;
      penalidadesAplicadas.push(p.id);
    });

    var multiplicador = 1;
    if (ge_indicatorMet_(ge_evidenceValue_(evidence, 'sem_infracoes')) && penalidadesAplicadas.length === 0) {
      multiplicador += 0.1;
    }
    if (ge_indicatorMet_(ge_evidenceValue_(evidence, 'explicou_com_as_proprias_palavras'))) {
      multiplicador += 0.05;
    }

    var total = Math.max(0, Math.round((base + bonus + penalidade) * multiplicador));
    return {
      success: true,
      data: {
        situacao: ge_clone_(situation),
        pontosBase: base,
        bonus: bonus,
        bonusAplicados: bonusAplicados,
        penalidade: penalidade,
        penalidadesAplicadas: penalidadesAplicadas,
        multiplicador: Math.round(multiplicador * 100) / 100,
        total: total,
        evidencias: evidence
      }
    };
  } catch (error) {
    Logger.log("Erro em calculateTrafficSituationScore: " + error.message);
    throw error;
  }
}

function awardTrafficSituation(alunoId, situationId, evidence, reason) {
  try {
    try {
      if (String(alunoId || '').trim() === '') return { success: false, message: 'alunoId obrigatorio.' };
      var score = calculateTrafficSituationScore(situationId, evidence || {});
      if (!score.success) return score;

      var data = score.data;
      var motivo = reason || ('Situação de trânsito: ' + data.situacao.nome);
      var eventRecord = wtgCreateRecord_(GAMIFICATION_EVENTS_SHEET, GAMIFICATION_EVENTS_HEADERS, {
        AlunoID: alunoId,
        SituacaoID: data.situacao.id,
        Categoria: data.situacao.categoria,
        PontosBase: data.pontosBase,
        Bonus: data.bonus,
        Penalidade: data.penalidade,
        Multiplicador: data.multiplicador,
        Total: data.total,
        Evidencias: JSON.stringify(data.evidencias || {}),
        Motivo: motivo
      }, { required: ['AlunoID', 'SituacaoID'] });

      var award = data.total > 0
        ? awardPoints(alunoId, data.total, motivo)
        : { success: true, data: { alunoId: alunoId, pontos: 0, total: getAlunoPoints(alunoId), distintivos: [] } };

      return {
        success: true,
        data: {
          alunoId: alunoId,
          situacao: data.situacao,
          pontuacao: data,
          evento: eventRecord.data,
          premio: award.data
        }
      };
    } catch (error) {
      Logger.log("Erro em awardTrafficSituation: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em awardTrafficSituation: " + error.message);
    throw error;
  }
}

function getTrafficSituationProgress(alunoId) {
  try {
    var rows = wtgReadObjects_(GAMIFICATION_EVENTS_SHEET).filter(function (r) {
      return String(r.AlunoID || r.alunoid || '') === String(alunoId);
    });
    var categorias = {};
    rows.forEach(function (r) {
      var cat = String(r.Categoria || r.categoria || 'geral');
      if (!categorias[cat]) categorias[cat] = { categoria: cat, eventos: 0, pontos: 0 };
      categorias[cat].eventos += 1;
      categorias[cat].pontos += Number(r.Total || r.total || 0);
    });
    return {
      totalEventos: rows.length,
      categorias: Object.keys(categorias).sort().map(function (cat) { return categorias[cat]; }),
      situacoesConcluidas: rows.map(function (r) { return String(r.SituacaoID || r.situacaoid || ''); })
        .filter(function (id, index, arr) { return id && arr.indexOf(id) === index; })
    };
  } catch (error) {
    Logger.log("Erro em getTrafficSituationProgress: " + error.message);
    throw error;
  }
}

function getTrafficGameSituationsForWeek(weekNumber) {
  try {
    var map = {
      1: ['faixa_entrequadra_108_109', 'area_escolar_superquadra'],
      2: ['quadra_vizinha_visitante', 'ponto_onibus_comercio_local'],
      3: ['eixinho_tesourinha_preferencia', 'rota_asa_norte_sul'],
      4: ['eixo_monumental_fluxo_civico', 'quadra_vizinha_visitante', 'rota_asa_norte_sul']
    };
    var ids = map[Number(weekNumber)] || [];
    return ge_clone_(TRAFFIC_GAME_SITUATIONS.filter(function (s) { return ids.indexOf(s.id) !== -1; }));
  } catch (error) {
    Logger.log("Erro em getTrafficGameSituationsForWeek: " + error.message);
    throw error;
  }
}

/** Garante a aba de pontos e retorna o total acumulado do aluno. */
function getAlunoPoints(alunoId) {
  try {
    if (String(alunoId || '').trim() === '') return 0;
    return wtgReadObjects_(PONTOS_SHEET)
      .filter(function (r) { return String(r.AlunoID || r.alunoid || '') === String(alunoId); })
      .reduce(function (s, r) { return s + (Number(r.Pontos) || 0); }, 0);
  } catch (error) {
    Logger.log("Erro em getAlunoPoints: " + error.message);
    throw error;
  }
}

/** Resolve o nível a partir de um total de pontos. */
function ge_levelForPoints_(total) {
  var current = GAMIFICATION_LEVELS[0];
  var next = null;
  for (var i = 0; i < GAMIFICATION_LEVELS.length; i++) {
    if (total >= GAMIFICATION_LEVELS[i].min) current = GAMIFICATION_LEVELS[i];
    else { next = GAMIFICATION_LEVELS[i]; break; }
  }
  return { current: current, next: next };
}

function awardPoints(alunoId, points, reason) {
  try {
    if (String(alunoId || '').trim() === '') return { success: false, message: 'alunoId obrigatorio.' };
    var pts = Number(points);
    if (isNaN(pts) || pts === 0) return { success: false, message: 'points deve ser um número diferente de zero.' };

    var antes = getAlunoPoints(alunoId);
    wtgCreateRecord_(PONTOS_SHEET, PONTOS_HEADERS, {
      AlunoID: alunoId, Pontos: pts, Motivo: reason || ''
    }, { required: ['AlunoID'] });
    var total = antes + pts;

    var nivelAntes = ge_levelForPoints_(antes).current.nivel;
    var nivelInfo = ge_levelForPoints_(total);
    var subiuNivel = nivelInfo.current.nivel > nivelAntes;

    if (typeof sendInAppNotification === 'function') {
      try {
        sendInAppNotification(alunoId, subiuNivel
          ? ('Parabéns! Você alcançou o nível "' + nivelInfo.current.nome + '".')
          : ('Você ganhou ' + pts + ' pontos: ' + (reason || 'atividade concluída') + '.'),
          subiuNivel ? 'nivel' : 'pontos');
      } catch (e) {}
    }

    // Atualiza distintivos por mérito (média de pontuações), se o módulo existir.
    var distintivos = [];
    if (typeof checkAndAwardAchievement === 'function') {
      try { distintivos = (checkAndAwardAchievement(alunoId).data || {}).awarded || []; } catch (e) {}
    }

    return {
      success: true,
      data: { alunoId: alunoId, pontos: pts, total: total, nivel: nivelInfo.current, subiuNivel: subiuNivel, distintivos: distintivos }
    };
  } catch (error) {
    Logger.log("Erro em awardPoints: " + error.message);
    throw error;
  }
}

function checkLevelUp(alunoId) {
  var total = getAlunoPoints(alunoId);
  var info = ge_levelForPoints_(total);
  var faltam = info.next ? Math.max(0, info.next.min - total) : 0;
  return { success: true, data: { total: total, nivel: info.current, proximo: info.next, pontosParaProximo: faltam } };
}

/** Recalcula o ranking de todos os alunos e persiste na aba Rankings. */
function updateLeaderboard() {
  try {
    var totais = {};
    wtgReadObjects_(PONTOS_SHEET).forEach(function (r) {
      var id = String(r.AlunoID || r.alunoid || '');
      if (!id) return;
      totais[id] = (totais[id] || 0) + (Number(r.Pontos) || 0);
    });

    var ranking = Object.keys(totais).map(function (id) {
      return { alunoId: id, total: totais[id], nivel: ge_levelForPoints_(totais[id]).current.nivel };
    }).sort(function (a, b) { return b.total - a.total; });

    ranking.forEach(function (item, i) {
      item.posicao = i + 1;
      var existente = wtgReadObjects_(RANKINGS_SHEET)
        .filter(function (r) { return String(r.AlunoID || r.alunoid || '') === item.alunoId; })[0];
      var payload = { AlunoID: item.alunoId, Total: item.total, Posicao: item.posicao, Nivel: item.nivel };
      if (existente && (existente.ID || existente.id)) wtgUpdateRecordById_(RANKINGS_SHEET, existente.ID || existente.id, payload);
      else wtgCreateRecord_(RANKINGS_SHEET, RANKINGS_HEADERS, payload, { required: ['AlunoID'] });
    });

    return { success: true, data: { ranking: ranking } };
  } catch (error) {
    Logger.log("Erro em updateLeaderboard: " + error.message);
    throw error;
  }
}

function getAlunoRank(alunoId) {
  try {
    var row = wtgReadObjects_(RANKINGS_SHEET)
      .filter(function (r) { return String(r.AlunoID || r.alunoid || '') === String(alunoId); })[0];
    if (row) return { success: true, data: { posicao: Number(row.Posicao) || null, total: Number(row.Total) || 0, nivel: Number(row.Nivel) || 1 } };
    // Sem ranking persistido: calcula na hora.
    var lb = updateLeaderboard().data.ranking;
    var found = lb.filter(function (r) { return r.alunoId === String(alunoId); })[0];
    return found
      ? { success: true, data: { posicao: found.posicao, total: found.total, nivel: found.nivel } }
      : { success: false, message: 'Aluno sem pontos registrados.' };
  } catch (error) {
    Logger.log("Erro em getAlunoRank: " + error.message);
    throw error;
  }
}

/** Perfil de gamificação consolidado para dashboards do aluno. */
function getGamificationProfile(alunoId) {
  var nivel = checkLevelUp(alunoId).data;
  var rank = getAlunoRank(alunoId);
  var distintivos = (typeof getAchievementsByAluno === 'function') ? getAchievementsByAluno(alunoId) : [];
  var progressoSituacoes = getTrafficSituationProgress(alunoId);
  return {
    success: true,
    data: {
      alunoId: alunoId,
      total: nivel.total,
      nivel: nivel.nivel,
      proximo: nivel.proximo,
      pontosParaProximo: nivel.pontosParaProximo,
      posicao: rank.success ? rank.data.posicao : null,
      distintivos: distintivos,
      progressoSituacoes: progressoSituacoes,
      situacoesDisponiveis: getTrafficGameSituations()
    }
  };
}

/** Registra os distintivos padrão de trânsito/cidadania (idempotente). */
function seedTrafficBadges() {
  try {
    if (typeof defineAchievement !== 'function') return { success: false, message: 'StudentAchievementManager indisponível.' };
    var existentes = wtgReadObjects_(CONQUISTAS_DEF_SHEET).map(function (d) { return String(d.ID || d.id || ''); });
    var criados = [];
    TRAFFIC_BADGES.forEach(function (b) {
      if (existentes.indexOf(b.id) !== -1) return;
      defineAchievement(b.id, b.criteria, b.descricao);
      criados.push(b.id);
    });
    return { success: true, data: { criados: criados } };
  } catch (error) {
    Logger.log("Erro em seedTrafficBadges: " + error.message);
    throw error;
  }
}
