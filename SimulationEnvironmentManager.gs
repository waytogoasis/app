// SimulationEnvironmentManager.gs
//
// Funcionalidade Principal: Gerencia a configuração e o estado do ambiente de simulação.
//
// Descrição: Define os parâmetros do ambiente de simulação (layout do pátio, sinais, elementos
//            interativos). Carrega diferentes configurações por ID; layout persistido em planilha
//            com um layout padrão embutido.
//
// Integrações:
// - Google Planilha (aba `AmbienteSimulacao`): Configurações do ambiente (Layout em JSON).
// - WeeklyDynamics.gs: ambiente apropriado por semana.
// - BrasiliaUrbanScaleContext.gs: cenário "superquadra" (acesso pela quadra vizinha).
//
// Funções Principais:
// - `loadEnvironment(environmentId)`: Carrega uma configuração de ambiente (ou o padrão).
// - `updateEnvironmentElement(elementId, newProperties)`: Atualiza propriedades de um elemento.
// - `getEnvironmentLayout()`: Retorna o layout atual do ambiente.
//
// Ambientes embutidos: `default` (pátio escolar genérico) e `superquadra` (Brasília, escala
// gregária — atravessar a superquadra vizinha para chegar à pretendida).

var AMBIENTE_SHEET = 'AmbienteSimulacao';
var AMBIENTE_HEADERS = ['ID', 'Layout', 'CriadoEm', 'AtualizadoEm'];
var AMBIENTE_DEFAULT_ID = 'default';
var AMBIENTE_DEFAULT_LAYOUT = {
  patio: { largura: 20, altura: 15 },
  elementos: [
    { id: 'semaforo_1', tipo: 'semaforo', x: 5, y: 5, estado: 'vermelho' },
    { id: 'faixa_1', tipo: 'faixa_pedestre', x: 10, y: 7 },
    { id: 'placa_pare', tipo: 'placa', x: 15, y: 3, texto: 'PARE' }
  ]
};

function sem_findRaw_(environmentId) {
  try {
    return wtgReadObjects_(AMBIENTE_SHEET)
      .filter(function (e) { return String(e.ID || e.id || '') === String(environmentId); })[0] || null;
  } catch (error) {
    Logger.log("Erro em sem_findRaw_: " + error.message);
    throw error;
  }
}

var AMBIENTE_SUPERQUADRA_ID = 'superquadra';

/** Layouts embutidos por ID (planilha tem prioridade sobre estes). */
function sem_builtinLayout_(environmentId) {
  try {
    if (environmentId === AMBIENTE_DEFAULT_ID) return JSON.parse(JSON.stringify(AMBIENTE_DEFAULT_LAYOUT));
    if (environmentId === AMBIENTE_SUPERQUADRA_ID && typeof getSuperquadraSimulationLayout === 'function') {
      return getSuperquadraSimulationLayout();
    }
    return null;
  } catch (error) {
    Logger.log("Erro em sem_builtinLayout_: " + error.message);
    throw error;
  }
}

function loadEnvironment(environmentId) {
  try {
    environmentId = environmentId || AMBIENTE_DEFAULT_ID;
    var raw = sem_findRaw_(environmentId);
    if (raw) { try { return { id: environmentId, layout: JSON.parse(raw.Layout || '{}') }; } catch (e) { return { id: environmentId, layout: {} }; } }
    var builtin = sem_builtinLayout_(environmentId);
    if (builtin) return { id: environmentId, layout: builtin };
    return null;
  } catch (error) {
    Logger.log("Erro em loadEnvironment: " + error.message);
    throw error;
  }
}

function getEnvironmentLayout() {
  return loadEnvironment(AMBIENTE_DEFAULT_ID).layout;
}

function sem_persistLayout_(environmentId, layout) {
  try {
    try {
      var raw = sem_findRaw_(environmentId);
      if (raw) return wtgUpdateRecordById_(AMBIENTE_SHEET, raw.ID, { Layout: JSON.stringify(layout) });
      return wtgCreateRecord_(AMBIENTE_SHEET, AMBIENTE_HEADERS, { ID: environmentId, Layout: JSON.stringify(layout) }, { required: ['ID'] });
    } catch (error) {
      Logger.log("Erro em sem_persistLayout_: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em sem_persistLayout_: " + error.message);
    throw error;
  }
}

function updateEnvironmentElement(elementId, newProperties) {
  try {
    var env = loadEnvironment(AMBIENTE_DEFAULT_ID);
    var layout = env.layout;
    var found = false;
    (layout.elementos || []).forEach(function (el) {
      if (String(el.id) === String(elementId)) {
        Object.keys(newProperties || {}).forEach(function (k) { el[k] = newProperties[k]; });
        found = true;
      }
    });
    if (!found) return { success: false, message: 'Elemento nao encontrado: ' + elementId };
    sem_persistLayout_(AMBIENTE_DEFAULT_ID, layout);
    return { success: true, data: { elementId: elementId, layout: layout } };
  } catch (error) {
    Logger.log("Erro em updateEnvironmentElement: " + error.message);
    throw error;
  }
}
