// PedagogicalContentManager.gs
//
// Funcionalidade Principal: Gerencia o conteúdo pedagógico e materiais didáticos do projeto.
//
// Descrição: Fornece acesso a planos de aula, descrições de atividades e materiais didáticos
//            do projeto "Brincando no Trânsito Seguro". Conteúdo embutido por padrão, sobreponível
//            pela aba `ConteudoPedagogico`.
//
// Integrações:
// - Google Planilha (aba `ConteudoPedagogico`): Conteúdo persistido/sobreposto.
// - WeeklyDynamics.gs: Conteúdo específico de cada semana.
//
// Funções Principais:
// - `getLessonPlan(weekNumber)`: Retorna o plano de aula para uma semana (1-4).
// - `getActivityDescription(activityId)`: Retorna a descrição de uma atividade.
// - `getEducationalMaterial(topic)`: Retorna materiais educativos sobre um tópico.

var CONTEUDO_SHEET = 'ConteudoPedagogico';
var LESSON_PLANS_DEFAULT = {
  1: {
    tema: 'Reconhecendo o trânsito na superquadra',
    objetivos: [
      'Identificar sinais, faixas, placas e sentidos de circulação',
      'Relacionar a regra ao lugar concreto: escola, bloco, pilotis, via interna e entrequadra',
      'Usar vocabulário básico de trânsito para explicar escolhas seguras'
    ]
  },
  2: {
    tema: 'O pedestre e a faixa em Brasília',
    objetivos: [
      'Reconhecer a faixa de pedestre como regra e como cultura de cidadania do DF',
      'Praticar olhar, esperar, sinalizar intenção e atravessar com segurança',
      'Diferenciar travessia segura em entrequadra, via interna e eixo/eixinho'
    ]
  },
  3: {
    tema: 'Sinalização, preferência e velocidade',
    objetivos: [
      'Reconhecer PARE, Dê a Preferência, semáforo, velocidade máxima e faixa',
      'Aplicar regras em cruzamentos, retornos, tesourinhas e acessos de superquadra',
      'Justificar a prioridade do pedestre e dos usuários vulneráveis'
    ]
  },
  4: {
    tema: 'Convivência e responsabilidade no trânsito de Brasília',
    objetivos: [
      'Explicar por que o maior cuida do menor no trânsito',
      'Agir como visitante ao atravessar a superquadra vizinha',
      'Registrar evidências de reconhecimento e aplicação das regras no cenário local'
    ]
  }
};

function pcm_findContent_(tipo, chave) {
  try {
    return wtgReadObjects_(CONTEUDO_SHEET).filter(function (c) {
      return String(c.Tipo || '') === tipo && String(c.Chave || '') === String(chave);
    })[0] || null;
  } catch (error) {
    Logger.log("Erro em pcm_findContent_: " + error.message);
    throw error;
  }
}

function getLessonPlan(weekNumber) {
  try {
    try {
      var custom = pcm_findContent_('plano', weekNumber);
      if (custom) { try { return JSON.parse(custom.Conteudo || '{}'); } catch (e) { return { conteudo: custom.Conteudo }; } }
      return LESSON_PLANS_DEFAULT[weekNumber] || null;
    } catch (error) {
      Logger.log("Erro em getLessonPlan: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em getLessonPlan: " + error.message);
    throw error;
  }
}

function getActivityDescription(activityId) {
  var custom = pcm_findContent_('atividade', activityId);
  if (custom) return { id: activityId, descricao: custom.Conteudo, fonte: 'planilha' };
  // Tenta a partir das atividades cadastradas (ActivityCrud), se houver.
  if (typeof getActivityById === 'function') {
    var act = getActivityById(activityId);
    if (act) return { id: activityId, descricao: act.prompt || act.Prompt || '', tipo: act.type || act.Tipo, fonte: 'atividades' };
  }
  return null;
}

function getEducationalMaterial(topic) {
  var custom = pcm_findContent_('material', topic);
  if (custom) return { topico: topic, material: custom.Conteudo, fonte: 'planilha' };
  // Tópicos de identidade do projeto: escalas de Brasília e psicologia do trânsito.
  var brasilia = pcm_brasiliaMaterial_(topic);
  if (brasilia) return { topico: topic, material: brasilia, fonte: 'brasilia' };
  return { topico: topic, material: 'Material educativo sobre ' + topic + ' (conteúdo padrão).', fonte: 'default' };
}

/** Material derivado do contexto urbano de Brasília, quando o módulo estiver presente. */
function pcm_brasiliaMaterial_(topic) {
  try {
    var t = String(topic || '').toLowerCase();
    if ((t.indexOf('regra') !== -1 || t.indexOf('sinal') !== -1 || t.indexOf('ctb') !== -1 || t.indexOf('marco legal') !== -1) &&
        typeof getBrasiliaTrafficRules === 'function') {
      return getBrasiliaTrafficRules().map(function (r) {
        return r.regra + ' Em Brasília: ' + r.contextoBrasilia + ' Como reconhecer: ' +
          r.reconhecer + ' Ação esperada: ' + r.comportamentoEsperado;
      }).join(' ');
    }
    if (t.indexOf('superquadra') !== -1 || t.indexOf('escala') !== -1 || t.indexOf('brasil') !== -1) {
      if (typeof getSuperquadraModel === 'function') {
        var m = getSuperquadraModel();
        return m.acessoVizinho.licaoCidadania + ' ' + m.acessoVizinho.licaoRegras +
          ' Exemplo: ' + m.exemplo + '. ' + m.acessoVizinho.dever;
      }
    }
    if ((t.indexOf('psicolog') !== -1 || t.indexOf('transito') !== -1 || t.indexOf('trânsito') !== -1) &&
        typeof getTrafficPsychologyNotes === 'function') {
      return getTrafficPsychologyNotes().map(function (n) { return n.tema + ': ' + n.nota; }).join(' ');
    }
    return null;
  } catch (error) {
    Logger.log("Erro em pcm_brasiliaMaterial_: " + error.message);
    throw error;
  }
}
