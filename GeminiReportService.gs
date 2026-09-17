/**
 * COMPONENTE: GeminiReportService.gs
 * PAPEL: Relatório pedagógico generativo via Google Gemini.
 *
 * Contrato de uso (frontend, google.script.run):
 *   - generateAiReport()        → relatório do panorama atual (tela inicial).
 *   - generateAiReport(payload) → relatório aprofundado; payload pode trazer
 *                                  { summary, foco, turma, periodo } vindos do clique.
 *
 * FONTE DE DADOS: getSchoolRealityAnalyticsSummary() (componente compartilhado
 * SchoolRealityAnalyticsService.gs, presente em toda a frota).
 *
 * RESILIÊNCIA (padrão da frota): retry + backoff exponencial em falhas
 * transitórias (HTTP 429/500/503 e exceções de rede). Sem GEMINI_API_KEY,
 * ou em erro permanente da API, degrada para um relatório local estruturado.
 *
 * AMBIENTE: propriedade de script GEMINI_API_KEY.
 */

var GeminiReportService = (function () {
  // FROTA-07: modelo lido da property do script, nunca hardcoded; cai no padrão local.
  function model_() {
    try {
      return PropertiesService.getScriptProperties().getProperty('GEMINI_MODEL') || 'gemini-2.0-flash';
    } catch (e) {
      return 'gemini-2.0-flash';
    }
  }
  var BASE_URL = 'https://generativelanguage.googleapis.com/v1beta/models/';
  var PROJECT = 'Way To Go As Is';
  var DOMAIN = 'simulações de cenários e decisões pedagógicas: padrões entre simulações e recomendações';
  var INSTRUCTION = 'Compare os resultados das simulações, identifique padrões recorrentes entre cenários e recomende próximos passos pedagógicos.';
  var INTERPRETATION_POLICY = {
    assessmentType: 'formative_contextualized',
    rubricRole: 'organizes_human_judgment',
    humanJudgmentRequired: true,
    claims: {
      exclusiveAssessment: false,
      stressReductionDemonstrated: false
    },
    note: 'Rubricas organizam o julgamento docente; não o eliminam. Hipóteses sobre estresse exigem avaliação específica.'
  };

  function apiKey_() {
    try { return PropertiesService.getScriptProperties().getProperty('GEMINI_API_KEY'); }
    catch (e) { return null; }
  }

  function isConfigured() { return !!apiKey_(); }

  function gatherSummary_() {
    try {
      if (typeof getSchoolRealityAnalyticsSummary === 'function') {
        return getSchoolRealityAnalyticsSummary({ record: false });
      }
    } catch (e) {}
    return { projectName: PROJECT, focus: DOMAIN, analyses: [] };
  }

  function analysesText_(summary) {
    var rows = (summary && summary.analyses) || [];
    if (!rows.length) return '(sem leituras registradas ainda)';
    return rows.map(function (a) {
      var val = (a.value === undefined || a.value === '') ? '—' : a.value;
      var unit = a.unit ? (' ' + a.unit) : '';
      var dim = a.dimension ? (' [' + a.dimension + ']') : '';
      var narr = a.narrative ? (' — ' + a.narrative) : '';
      return '- ' + (a.label || a.key) + ': ' + val + unit + dim + narr;
    }).join('\n');
  }

  function buildPrompt_(summary, payload) {
    try {
      var foco = (payload && payload.foco) || (summary && summary.focus) || DOMAIN;
      var escopo = (payload && payload.turma) ? ('Turma/recorte: ' + payload.turma + '.') : 'Recorte: visão geral da escola.';
      return [
        'Você é um analista pedagógico da Escola Classe 115 Norte (SEEDF, Brasília-DF).',
        'Projeto: ' + PROJECT + '. Domínio: ' + DOMAIN + '.',
        escopo,
        'Foco da análise: ' + foco + '.',
        '',
        'Leituras (indicadores de realidade escolar):',
        analysesText_(summary),
        '',
        INSTRUCTION,
        '',
        'Política de interpretação: trate os resultados como evidências formativas e contextuais.',
        'A rubrica organiza o julgamento humano do professor, mas não o substitui nem constitui avaliação exclusiva.',
        'Não afirme baixo estresse, redução de estresse ou ausência de estresse: isso não foi demonstrado pelos dados.',
        'Não transforme padrões de simulação em diagnóstico clínico, traço estável ou relação causal.',
        '',
        'Produza um relatório em português do Brasil, objetivo e acionável, com as seções:',
        '1. Panorama (2–3 frases).',
        '2. Destaques positivos.',
        '3. Pontos de atenção.',
        '4. Recomendações práticas para o professor/gestor (até 5 itens).',
        'Não invente números; use apenas os dados fornecidos.'
      ].join('\n');
    } catch (error) {
      Logger.log("Erro em buildPrompt_: " + error.message);
      throw error;
    }
  }

  function hasUnsupportedAssessmentClaim_(text) {
    var normalized = String(text || '').toLowerCase();
    return /baixo estresse|redu[cç][aã]o de estresse|sem estresse|estresse reduzido|avalia[cç][aã]o exclusiva|exclusivamente objetiva|elimina o julgamento|substitui o professor|substitui a avalia[cç][aã]o humana/.test(normalized);
  }

  function interpretationPolicy_() {
    return {
      assessmentType: INTERPRETATION_POLICY.assessmentType,
      rubricRole: INTERPRETATION_POLICY.rubricRole,
      humanJudgmentRequired: INTERPRETATION_POLICY.humanJudgmentRequired,
      claims: {
        exclusiveAssessment: INTERPRETATION_POLICY.claims.exclusiveAssessment,
        stressReductionDemonstrated: INTERPRETATION_POLICY.claims.stressReductionDemonstrated
      },
      note: INTERPRETATION_POLICY.note
    };
  }

  function fetchGeminiReport_(url, options) {
    var opts = {};
    for (var k in options) { if (Object.prototype.hasOwnProperty.call(options, k)) opts[k] = options[k]; }
    opts.muteHttpExceptions = true;

    var MAX = 3;
    var waitMs = 700;
    for (var attempt = 1; attempt <= MAX; attempt++) {
      var resp;
      try {
        resp = UrlFetchApp.fetch(url, opts);
      } catch (e) {
        if (attempt >= MAX) throw e;
        Utilities.sleep(waitMs); waitMs *= 2; continue;
      }
      var code = resp.getResponseCode();
      var transient = (code === 429 || code === 500 || code === 503);
      if (transient && attempt < MAX) { Utilities.sleep(waitMs); waitMs *= 2; continue; }
      if (code >= 400) throw new Error('Gemini HTTP ' + code + ': ' + resp.getContentText().slice(0, 300));
      return resp;
    }
  }

  function callGemini_(prompt) {
    try {
      var key = apiKey_();
      var url = BASE_URL + model_() + ':generateContent?key=' + encodeURIComponent(key);
      var options = {
        method: 'post',
        contentType: 'application/json',
        payload: JSON.stringify({
          contents: [{ parts: [{ text: prompt }] }],
          generationConfig: { temperature: 0.4, maxOutputTokens: 1200 }
        })
      };
      var resp = fetchGeminiReport_(url, options);
      var data = JSON.parse(resp.getContentText());
      // FROTA-XX: extração defensiva com GeminiResponseNormalizer
      var text = GeminiResponseNormalizer.extractText(data);
      if (!text) throw new Error('Resposta vazia do Gemini.');
      return String(text).trim();
    } catch (error) {
      Logger.log("Erro em callGemini_: " + error.message);
      throw error;
    }
  }

  function fallback_(summary) {
    var head = 'Relatório local (' + PROJECT + ') — IA indisponível (configure GEMINI_API_KEY para a versão completa).';
    return head + '\n\nPanorama — foco: ' + DOMAIN + '.\n\nLeituras:\n' + analysesText_(summary) +
      '\n\nRecomendações gerais:\n' +
      '- Revisar os indicadores em destaque com a turma.\n' +
      '- Priorizar os pontos de atenção com maior impacto no engajamento.\n' +
      '- Registrar novas observações para enriquecer a próxima análise.\n\n' +
      'Nota de interpretação: esta é uma avaliação formativa contextualizada. A rubrica organiza o julgamento docente, não o elimina; qualquer hipótese de redução de estresse requer avaliação específica.';
  }

  function generate(payload) {
    try {
      payload = payload || {};
      var summary = payload.summary || gatherSummary_();
      var base = {
        success: true,
        project: PROJECT,
        generatedAt: new Date(),
        summary: summary,
        interpretation: interpretationPolicy_()
      };
      if (!isConfigured()) {
        base.source = 'fallback'; base.report = fallback_(summary); return base;
      }
      try {
        var report = callGemini_(buildPrompt_(summary, payload));
        if (hasUnsupportedAssessmentClaim_(report)) {
          throw new Error('Resposta gerada contém uma alegação de avaliação não demonstrada.');
        }
        base.source = 'gemini'; base.report = report; return HumanReviewService.decorateResult('ai.report', base, base.report);
      } catch (e) {
        base.source = 'fallback'; base.report = fallback_(summary);
        base.error = String((e && e.message) || e); return base;
      }
    } catch (error) {
      Logger.log("Erro em generate: " + error.message);
      throw error;
    }
  }

  return { isConfigured: isConfigured, generate: generate };
})();

/**
 * Ponto de entrada para o frontend (google.script.run.generateAiReport).
 * @param {Object=} payload { summary?, foco?, turma?, periodo? }
 * @return {{success:boolean, source:string, project:string, report:string, summary:Object}}
 */
function generateAiReport(payload) {
  return GeminiReportService.generate(payload || {});
}

