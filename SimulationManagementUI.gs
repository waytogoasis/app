// SimulationManagementUI.gs
//
// Funcionalidade Principal: Funções de Apps Script para interagir com a interface de gerenciamento de simulações.
//
// Descrição: Este script atua como uma ponte entre o frontend HTML de gerenciamento de simulações
//            e o backend `SimulacaoService.gs`. Ele recebe requisições da UI, chama as funções
//            apropriadas do `SimulacaoService.gs` e retorna os resultados para a interface.
//
// Integrações:
// - SimulacaoService.gs: Para realizar operações CRUD de simulações.
// - HtmlService.gs: Para servir a página `SimulationOverview.html`.
// - PermissionService.gs: Para verificar permissões antes de executar ações.
//
// Funções Principais:
// - `getSimulationsForUI()`: Retorna uma lista de simulações para exibição na UI.
// - `startSimulationFromUI(simulationData)`: Inicia uma nova simulação a partir dos dados da UI.
// - `endSimulationFromUI(simulationId, finalObservacoes)`: Finaliza uma simulação a partir da UI.
// - `deleteSimulationFromUI(simulationId)`: Deleta uma simulação a partir da UI.
//
// Observações: Garante que as interações da interface do usuário com o backend sejam seguras e eficientes.

function getSimulationsForUI() {
  throw new Error('Bridge legado desativado. Use apiCall("Simulations", "list", payload).');
}

function getAllSimulationsForUI() {
  return getSimulationsForUI();
}

function startSimulationFromUI(simulationData) {
  throw new Error('Bridge legado desativado. Use apiCall("Simulations", "create", payload).');
}

function endSimulationFromUI(simulationId, finalObservacoes) {
  throw new Error('Bridge legado desativado. Use apiCall("Simulations", "finish", payload).');
}

function deleteSimulationFromUI(simulationId) {
  throw new Error('Exclusao de simulacao nao esta exposta no contrato atual.');
}
