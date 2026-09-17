// SimulationDataProcessor.gs
//
// Funcionalidade Principal: Processa e normaliza dados brutos de simulações.
//
// Descrição: Limpa, valida e transforma os dados de entrada relacionados às simulações de trânsito,
//            garantindo formato consistente antes do armazenamento.
//
// Integrações:
// - SimulacaoService.gs: Utiliza para persistir os dados processados.
// - ValidationUtils.gs: Para validar a integridade dos dados.
//
// Funções Principais:
// - `processNewSimulationData(rawData)`: Limpa e valida dados de uma nova simulação.
// - `normalizeSimulationType(type)`: Normaliza o tipo de simulação para um vocabulário controlado.
// - `formatSimulationDate(date)`: Formata a data da simulação (ISO yyyy-MM-dd).

var SIMULATION_TYPE_MAP = {
  baliza: 'baliza', estacionamento: 'baliza',
  faixa: 'faixa_pedestre', pedestre: 'faixa_pedestre', 'faixa_pedestre': 'faixa_pedestre',
  semaforo: 'semaforo', sinaleiro: 'semaforo',
  rotatoria: 'rotatoria', rotunda: 'rotatoria',
  livre: 'livre', padrao: 'padrao'
};

function normalizeSimulationType(type) {
  try {
    var key = String(type || '').trim().toLowerCase().replace(/\s+/g, '_').replace(/[áàâã]/g, 'a').replace(/[éê]/g, 'e').replace(/[í]/g, 'i').replace(/[óôõ]/g, 'o').replace(/[ú]/g, 'u');
    return SIMULATION_TYPE_MAP[key] || (key || 'padrao');
  } catch (error) {
    Logger.log("Erro em normalizeSimulationType: " + error.message);
    throw error;
  }
}

function formatSimulationDate(date) {
  try {
    if (!date) return new Date().toISOString().slice(0, 10);
    var d = new Date(date);
    if (isNaN(d.getTime())) return '';
    return d.toISOString().slice(0, 10);
  } catch (error) {
    Logger.log("Erro em formatSimulationDate: " + error.message);
    throw error;
  }
}

function processNewSimulationData(rawData) {
  try {
    rawData = rawData || {};
    var alunoId = rawData.alunoId || rawData.AlunoID;
    if (typeof isNotNullOrEmpty === 'function' ? !isNotNullOrEmpty(alunoId) : !alunoId) {
      return { success: false, errors: ['alunoId obrigatorio.'] };
    }
    return {
      success: true,
      data: {
        AlunoID: alunoId,
        Tipo: normalizeSimulationType(rawData.tipo || rawData.Tipo || rawData.tipoSimulacao),
        Data: formatSimulationDate(rawData.data || rawData.Data),
        Observacoes: String(rawData.observacoes || rawData.Observacoes || '').trim()
      }
    };
  } catch (error) {
    Logger.log("Erro em processNewSimulationData: " + error.message);
    throw error;
  }
}
