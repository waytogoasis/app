// PedagogicalEval.gs
//
// Funcionalidade Principal: Funções específicas para avaliação pedagógica geral do projeto.
//
// Descrição: Este script contém a lógica para avaliar e registrar o desempenho dos alunos
//            em relação aos objetivos pedagógicos gerais do projeto, incluindo a formação
//            de hábitos seguros, argumentação moral e a transferência do aprendizado para
//            o cotidiano. Complementa as avaliações mais específicas de psicomotricidade,
//            cognição e funções executivas.
//
// Integrações:
// - Google Planilha (aba `Aval_<Dimensao>`): Armazenamento dos resultados por dimensão.
// - UserService.gs (wtg* helpers): camada de CRUD reutilizável.
// - SimulacaoService.gs: associa a avaliação a uma simulação específica.
//
// Funções Principais:
// - `evaluatePedagogicalGoals(simulacaoId, alunoId, data)`: Avalia e registra o cumprimento dos objetivos pedagógicos.
// - `getPedagogicalScores(alunoId)`: Retorna as pontuações pedagógicas de um aluno.
// - `analyzePedagogicalTrends(alunoId)`: Analisa tendências de desempenho pedagógico.
//
// Observações: Hospeda os helpers `wtgEval*_` compartilhados por todos os módulos *Eval.

// ===== Helpers compartilhados de avaliação por dimensão (usados por todos os *Eval) =====
var WTG_EVAL_HEADERS = ['ID', 'SimulacaoID', 'AlunoID', 'Dimensao', 'Indicadores', 'Score', 'CriadoEm', 'AtualizadoEm'];

// Contrato semântico da avaliação: a rubrica estrutura evidências observáveis,
// mas não transforma o resultado em medida automática nem substitui o professor.
var WTG_ASSESSMENT_INTERPRETATION = {
  assessmentType: 'formative_contextualized',
  rubricRole: 'organizes_human_judgment',
  humanJudgmentRequired: true,
  claims: {
    exclusiveAssessment: false,
    stressReductionDemonstrated: false
  },
  note: 'A rubrica organiza o julgamento docente; resultados são evidências parciais e contextuais.'
};

function wtgAssessmentInterpretation_() {
  return {
    assessmentType: WTG_ASSESSMENT_INTERPRETATION.assessmentType,
    rubricRole: WTG_ASSESSMENT_INTERPRETATION.rubricRole,
    humanJudgmentRequired: WTG_ASSESSMENT_INTERPRETATION.humanJudgmentRequired,
    claims: {
      exclusiveAssessment: WTG_ASSESSMENT_INTERPRETATION.claims.exclusiveAssessment,
      stressReductionDemonstrated: WTG_ASSESSMENT_INTERPRETATION.claims.stressReductionDemonstrated
    },
    note: WTG_ASSESSMENT_INTERPRETATION.note
  };
}

function wtgEvalSheet_(dimensao) { return 'Aval_' + String(dimensao).replace(/[^A-Za-z0-9]+/g, '_'); }

// ---------------------------------------------------------------------------
// ESCALAS DE AFERIÇÃO — âncoras comportamentais por indicador
// Cada indicador tem 5 níveis (0, 25, 50, 75, 100) com descrição observável.
// O professor escolhe o nível que melhor descreve o comportamento observado.
// ---------------------------------------------------------------------------
var WTG_RUBRICA_ESCALAS = {
  // Cognitiva (Piaget)
  compreensao_regras: {
    label: 'Compreensão de Regras de Trânsito',
    descricao: 'Capacidade de identificar e explicar regras de circulação observadas na simulação.',
    niveis: {
      0:   'Não reconhece nenhuma regra; ações aleatórias.',
      25:  'Reconhece 1 regra somente quando lembrada; não generaliza.',
      50:  'Identifica 2–3 regras; aplica com apoio verbal do professor.',
      75:  'Identifica e aplica regras corretamente sem auxílio; explica com vocabulário próprio.',
      100: 'Identifica, aplica e justifica regras em situações novas; transfere para contexto real.'
    }
  },
  reversibilidade: {
    label: 'Reversibilidade (Piaget)',
    descricao: 'Capacidade de compreender que uma ação de trânsito pode ser revertida ou que regras são bidirecionais.',
    niveis: {
      0:   'Não percebe reversibilidade; trata cada situação como única e definitiva.',
      25:  'Percebe retorno físico (pode voltar) mas não compreende simetria de regras.',
      50:  'Compreende que o pedestre e o motorista têm deveres simétricos com apoio.',
      75:  'Demonstra espontaneamente que regras funcionam em ambos os sentidos.',
      100: 'Aplica reversibilidade para resolver conflitos de trânsito de forma autônoma.'
    }
  },
  descentracao: {
    label: 'Descentração (Piaget)',
    descricao: 'Capacidade de considerar o ponto de vista dos outros usuários da via.',
    niveis: {
      0:   'Age exclusivamente do próprio ponto de vista; ignora outros usuários.',
      25:  'Nota a presença dos outros mas não considera suas necessidades.',
      50:  'Nomeia o ponto de vista de outro usuário quando questionado.',
      75:  'Demonstra empatia espontânea; ajusta comportamento para não prejudicar outros.',
      100: 'Negocia e propõe soluções que beneficiam múltiplos usuários da via.'
    }
  },
  // Funções Executivas
  controle_inibitorio: {
    label: 'Controle Inibitório',
    descricao: 'Capacidade de inibir impulsos e aguardar sinal/vez correta na simulação.',
    niveis: {
      0:   'Não aguarda; age impulsivamente em todas as situações.',
      25:  'Aguarda apenas após intervenção direta do professor.',
      50:  'Aguarda na maioria das vezes; falha em situações de maior excitação.',
      75:  'Controla impulsos de forma consistente; avisa quando quer agir.',
      100: 'Controla impulsos e auxilia colegas a fazerem o mesmo.'
    }
  },
  memoria_trabalho: {
    label: 'Memória de Trabalho',
    descricao: 'Capacidade de lembrar regras e sequências durante a simulação sem consulta.',
    niveis: {
      0:   'Não retém nenhuma regra após explicação.',
      25:  'Lembra 1 regra por vez; esquece ao mudar de situação.',
      50:  'Retém 2–3 regras durante a simulação com lembretes ocasionais.',
      75:  'Lembra as regras principais e as aplica em sequência correta.',
      100: 'Recorre à memória para situações complexas sem apoio.'
    }
  },
  flexibilidade_cognitiva: {
    label: 'Flexibilidade Cognitiva',
    descricao: 'Capacidade de adaptar comportamento quando as regras da simulação mudam.',
    niveis: {
      0:   'Insiste no padrão anterior mesmo após nova instrução.',
      25:  'Muda somente com múltiplas repetições da nova instrução.',
      50:  'Adapta-se após 1 repetição com alguma hesitação.',
      75:  'Adapta-se rapidamente; verbaliza a mudança de regra.',
      100: 'Antecipa mudanças e propõe adaptações para o grupo.'
    }
  },
  // Psicomotricidade
  coordenacao_motora: {
    label: 'Coordenação Motora',
    descricao: 'Qualidade do controle do corpo/veículo-brinquedo na simulação espacial.',
    niveis: {
      0:   'Movimento descontrolado; colisões frequentes.',
      25:  'Movimento controlado em linha reta; dificuldade em curvas.',
      50:  'Curvas e paradas com qualidade razoável; colisões esporádicas.',
      75:  'Controle fluido em trajetórias variadas; ajusta velocidade.',
      100: 'Controle preciso; demonstra para colegas e corrige próprias trajetórias.'
    }
  },
  // Pedagógica
  habitos_seguros: {
    label: 'Hábitos Seguros de Trânsito',
    descricao: 'Comportamentos de segurança observados espontaneamente na simulação.',
    niveis: {
      0:   'Nenhum comportamento seguro observado.',
      25:  'Demonstra 1 hábito somente quando solicitado.',
      50:  'Demonstra 2–3 hábitos; varia conforme atenção do professor.',
      75:  'Demonstra hábitos de forma consistente e autônoma.',
      100: 'Demonstra hábitos e os nomeia/justifica; influencia colegas positivamente.'
    }
  },
  argumentacao_moral: {
    label: 'Argumentação Moral sobre Trânsito',
    descricao: 'Qualidade dos argumentos sobre deveres e ética no trânsito nos debates.',
    niveis: {
      0:   'Não participa ou repete falas sem sentido.',
      25:  'Apresenta opinião sem justificativa.',
      50:  'Justifica com base em consequência imediata ("porque bate").',
      75:  'Argumenta com base em dever e direito ("é obrigação do motorista").',
      100: 'Articula argumentos usando norma, consequência e responsabilidade; rebate contrapontos.'
    }
  }
};

/**
 * Retorna o mapa completo de escalas de aferição para uso no frontend.
 * @return {Object}
 */
function wtgGetEscalasAferio() {
  return WTG_RUBRICA_ESCALAS;
}

/**
 * Retorna a âncora comportamental para um indicador e nível específicos.
 * @param {string} indicador
 * @param {number} nivel  0, 25, 50, 75 ou 100
 * @return {string}
 */
function wtgGetAncora(indicador, nivel) {
  try {
    var escala = WTG_RUBRICA_ESCALAS[indicador];
    if (!escala) return '(indicador não catalogado)';
    // Mapeia para o nível mais próximo disponível (0, 25, 50, 75, 100)
    var nivelArred = Math.round(Number(nivel) / 25) * 25;
    nivelArred = Math.max(0, Math.min(100, nivelArred));
    return escala.niveis[nivelArred] || '(nível não definido)';
  } catch (error) {
    Logger.log("Erro em wtgGetAncora: " + error.message);
    throw error;
  }
}

/**
 * Valida se um score é consistente com as âncoras da escala (múltiplo de 25).
 * Retorna o score arredondado e a âncora correspondente.
 * @param {string} indicador
 * @param {number} scoreRaw  Valor bruto inserido pelo professor (0–100).
 * @return {{ score: number, ancora: string, arredondado: boolean }}
 */
function wtgValidarScore(indicador, scoreRaw) {
  try {
    var val = Math.max(0, Math.min(100, Number(scoreRaw) || 0));
    var arredondado = val % 25 !== 0;
    var scoreNorm   = Math.round(val / 25) * 25;
    return {
      score: scoreNorm,
      ancora: wtgGetAncora(indicador, scoreNorm),
      arredondado: arredondado,
      valorOriginal: val
    };
  } catch (error) {
    Logger.log("Erro em wtgValidarScore: " + error.message);
    throw error;
  }
}

// ---------------------------------------------------------------------------
// SISTEMA UNIFICADO DE PONTUAÇÃO
// A pontuação ponderada é uma síntese descritiva de evidências registradas;
// a interpretação pedagógica continua dependendo do julgamento docente.
// ---------------------------------------------------------------------------

/**
 * Pesos padrão por indicador dentro de cada dimensão.
 * A ausência de um indicador na tabela implica peso 1 (tratamento igual).
 */
var WTG_PESOS_INDICADORES = {
  // Cognitiva
  compreensao_regras:     1.5,
  reversibilidade:        1.0,
  descentracao:           1.0,
  // Funções Executivas
  controle_inibitorio:    1.5,
  memoria_trabalho:       1.0,
  flexibilidade_cognitiva: 1.0,
  // Psicomotricidade
  coordenacao_motora:     1.0,
  // Pedagógica
  habitos_seguros:        1.5,
  argumentacao_moral:     1.0
};

/**
 * Calcula o score ponderado de uma avaliação.
 * Usa os pesos de WTG_PESOS_INDICADORES; indicadores sem peso recebem peso 1.
 * Substitui wtgEvalScore_ (média simples) e applyRubrica (ponderada isolada).
 *
 * @param {Object} data  Mapa indicador → valor numérico (0–100).
 * @return {number}  Score ponderado arredondado a 2 casas decimais (0–100).
 */
function wtgEvalScore_(data) {
  try {
    if (!data || typeof data !== 'object') return 0;
    var somaPesos  = 0;
    var somaValores = 0;
    Object.keys(data).forEach(function (k) {
      var val = Number(data[k]);
      if (isNaN(val)) return;
      var peso = WTG_PESOS_INDICADORES[k] || 1;
      somaValores += val * peso;
      somaPesos   += peso;
    });
    if (!somaPesos) return 0;
    return Math.round((somaValores / somaPesos) * 100) / 100;
  } catch (error) {
    Logger.log("Erro em wtgEvalScore_: " + error.message);
    throw error;
  }
}

function wtgEvalRecord_(dimensao, simulacaoId, alunoId, data) {
  try {
    if (String(alunoId || '').trim() === '') return { success: false, message: 'alunoId obrigatorio.' };
    var score = wtgEvalScore_(data);
    var result = wtgCreateRecord_(wtgEvalSheet_(dimensao), WTG_EVAL_HEADERS, {
      SimulacaoID: simulacaoId || '',
      AlunoID: alunoId,
      Dimensao: dimensao,
      Indicadores: JSON.stringify(data || {}),
      Score: score
    }, { required: ['AlunoID'] });
    if (result.success) {
      result.data.Score = score;
      result.data.assessmentInterpretation = wtgAssessmentInterpretation_();
    }
    return result;
  } catch (error) {
    Logger.log("Erro em wtgEvalRecord_: " + error.message);
    throw error;
  }
}

function wtgEvalScores_(dimensao, alunoId) {
  try {
    return wtgReadObjects_(wtgEvalSheet_(dimensao))
      .filter(function (r) { return String(r.AlunoID || r.alunoid || '') === String(alunoId); })
      .map(function (r) {
        try { r.IndicadoresParsed = JSON.parse(r.Indicadores || '{}'); } catch (e) { r.IndicadoresParsed = {}; }
        r.Score = Number(r.Score) || 0;
        r.assessmentInterpretation = wtgAssessmentInterpretation_();
        return r;
      });
  } catch (error) {
    Logger.log("Erro em wtgEvalScores_: " + error.message);
    throw error;
  }
}

function wtgEvalTrends_(dimensao, alunoId) {
  var scores = wtgEvalScores_(dimensao, alunoId)
    .sort(function (a, b) { return new Date(a.CriadoEm) - new Date(b.CriadoEm); });
  if (!scores.length) return {
    dimensao: dimensao,
    avaliacoes: 0,
    tendencia: 'sem_dados',
    media: 0,
    delta: 0,
    assessmentInterpretation: wtgAssessmentInterpretation_()
  };
  var primeira = scores[0].Score, ultima = scores[scores.length - 1].Score;
  var delta = Math.round((ultima - primeira) * 100) / 100;
  var tendencia = delta > 2 ? 'melhorando' : (delta < -2 ? 'declinando' : 'estavel');
  var media = Math.round(scores.reduce(function (s, x) { return s + x.Score; }, 0) / scores.length * 100) / 100;
  return {
    dimensao: dimensao, avaliacoes: scores.length, tendencia: tendencia,
    media: media, delta: delta, primeira: primeira, ultima: ultima,
    assessmentInterpretation: wtgAssessmentInterpretation_(),
    serie: scores.map(function (s) { return { data: s.CriadoEm, score: s.Score }; })
  };
}

// ===== Avaliação pedagógica geral =====
// Indicadores sugeridos: habitos_seguros, argumentacao_moral, transferencia_cotidiano (0-100).
var DIM_PEDAGOGICA = 'Pedagogica';

function evaluatePedagogicalGoals(simulacaoId, alunoId, data) {
  return wtgEvalRecord_(DIM_PEDAGOGICA, simulacaoId, alunoId, data);
}

function getPedagogicalScores(alunoId) {
  return wtgEvalScores_(DIM_PEDAGOGICA, alunoId);
}

function analyzePedagogicalTrends(alunoId) {
  return wtgEvalTrends_(DIM_PEDAGOGICA, alunoId);
}
