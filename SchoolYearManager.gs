// SchoolYearManager.gs
//
// Funcionalidade Principal: Gerencia os anos letivos e seus períodos no sistema.
//
// Descrição: Permite definir e gerenciar os anos letivos (datas de início/fim) e associar
//            turmas a períodos específicos, apoiando o acompanhamento do progresso ao longo do tempo.
//
// Integrações:
// - Google Planilha (aba `AnosLetivos`): Armazenamento dos anos letivos.
// - ClassroomManager.gs: Para associar turmas a anos letivos.
// - UserService.gs (wtg* helpers): camada de CRUD reutilizável.
//
// Funções Principais:
// - `createSchoolYear(yearData)`: Cria um novo ano letivo.
// - `getCurrentSchoolYear()`: Retorna o ano letivo marcado como atual (ou que contém a data de hoje).
// - `getSchoolYears()`: Retorna todos os anos letivos registrados.
// - `associateClassroomToYear(classId, yearId)`: Associa uma turma a um ano letivo.

var ANOS_LETIVOS_SHEET = 'AnosLetivos';
var ANOS_LETIVOS_HEADERS = ['ID', 'Nome', 'Inicio', 'Fim', 'Atual', 'CriadoEm', 'AtualizadoEm'];

function createSchoolYear(yearData) {
  try {
    yearData = yearData || {};
    if (String(yearData.nome || yearData.Nome || '').trim() === '') return { success: false, message: 'Nome do ano letivo obrigatorio.' };
    return wtgCreateRecord_(ANOS_LETIVOS_SHEET, ANOS_LETIVOS_HEADERS, {
      Nome: yearData.nome || yearData.Nome,
      Inicio: yearData.inicio || yearData.Inicio || '',
      Fim: yearData.fim || yearData.Fim || '',
      Atual: yearData.atual === true || yearData.Atual === true
    }, { required: ['Nome'] });
  } catch (error) {
    Logger.log("Erro em createSchoolYear: " + error.message);
    throw error;
  }
}

function getCurrentSchoolYear() {
  try {
    var anos = wtgReadObjects_(ANOS_LETIVOS_SHEET);
    var marcado = anos.filter(function (a) { return a.Atual === true || String(a.Atual).toLowerCase() === 'true'; })[0];
    if (marcado) return marcado;
    var hoje = new Date();
    return anos.filter(function (a) {
      var ini = a.Inicio ? new Date(a.Inicio) : null;
      var fim = a.Fim ? new Date(a.Fim) : null;
      return (!ini || ini <= hoje) && (!fim || fim >= hoje);
    })[0] || null;
  } catch (error) {
    Logger.log("Erro em getCurrentSchoolYear: " + error.message);
    throw error;
  }
}

function getSchoolYears() {
  return wtgReadObjects_(ANOS_LETIVOS_SHEET);
}

function associateClassroomToYear(classId, yearId) {
  return wtgUpdateRecordById_(TURMAS_SHEET, classId, { AnoLetivoID: yearId });
}
