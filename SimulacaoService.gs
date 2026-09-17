// SimulacaoService.gs
//
// Funcionalidade Principal: Gerencia o ciclo de vida das simulações de trânsito.
//
// Descrição: Este script é responsável por registrar o início e o fim das sessões de simulação,
//            associando-as a alunos específicos e registrando informações relevantes como data,
//            tipo de simulação e observações. Ele interage com a aba 'Simulacoes' da Google Planilha.
//
// Integrações:
// - Google Planilha (aba 'Simulacoes'): Todas as operações de dados são realizadas nesta aba.
// - UserService.gs (wtg* helpers): camada de CRUD reutilizável.
// - AlunoService.gs: valida a existência do aluno antes de registrar uma simulação.
//
// Funções Principais:
// - `startSimulation(alunoId, tipoSimulacao, observacoes)`: Registra o início de uma nova simulação.
// - `endSimulation(simulationId, finalObservacoes)`: Finaliza uma simulação existente.
// - `getSimulationById(simulationId)`: Retorna os dados de uma simulação específica.
// - `getSimulationsByAluno(alunoId)`: Retorna todas as simulações de um aluno.
// - `getAllSimulations()`: Retorna uma lista de todas as simulações registradas.

var SIMULACOES_SHEET = 'Simulacoes';
var SIMULACOES_HEADERS = ['ID', 'AlunoID', 'Tipo', 'Status', 'Observacoes', 'IniciadoEm', 'FinalizadoEm', 'CriadoEm', 'AtualizadoEm'];

function startSimulation(alunoId, tipoSimulacao, observacoes, dataSimulacao) {
  try {
    if (String(alunoId || '').trim() === '') return { success: false, message: 'alunoId obrigatorio.' };
    if (typeof getAlunoById === 'function' && String(alunoId).charAt(0) !== '[') {
      var aluno = getAlunoById(alunoId);
      if (aluno && aluno.success === false) return { success: false, message: 'Aluno inexistente.' };
    }
    return wtgCreateRecord_(SIMULACOES_SHEET, SIMULACOES_HEADERS, {
      AlunoID: alunoId,
      Tipo: tipoSimulacao || 'padrao',
      Status: 'em_andamento',
      Observacoes: observacoes || '',
      IniciadoEm: dataSimulacao || new Date().toISOString(),
      FinalizadoEm: ''
    }, { required: ['AlunoID'] });
  } catch (error) {
    Logger.log("Erro em startSimulation: " + error.message);
    throw error;
  }
}

function endSimulation(simulationId, finalObservacoes) {
  try {
    var current = wtgFindRecordById_(SIMULACOES_SHEET, simulationId);
    if (!current.success) return { success: false, message: 'Simulacao nao encontrada.' };
    var updates = { Status: 'finalizada', FinalizadoEm: new Date().toISOString() };
    if (finalObservacoes !== undefined && finalObservacoes !== null && String(finalObservacoes) !== '') {
      var prev = current.data.Observacoes || '';
      updates.Observacoes = prev ? (prev + ' | ' + finalObservacoes) : finalObservacoes;
    }
    return wtgUpdateRecordById_(SIMULACOES_SHEET, simulationId, updates);
  } catch (error) {
    Logger.log("Erro em endSimulation: " + error.message);
    throw error;
  }
}

function getSimulationById(simulationId) {
  return wtgFindRecordById_(SIMULACOES_SHEET, simulationId);
}

function getSimulationsByAluno(alunoId) {
  try {
    return wtgReadObjects_(SIMULACOES_SHEET).filter(function (s) {
      return String(s.AlunoID || s.alunoid || '') === String(alunoId);
    });
  } catch (error) {
    Logger.log("Erro em getSimulationsByAluno: " + error.message);
    throw error;
  }
}

function getAllSimulations() {
  return wtgReadObjects_(SIMULACOES_SHEET);
}
