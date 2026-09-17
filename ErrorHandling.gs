// ErrorHandling.gs
//
// Funcionalidade Principal: Gerencia o tratamento de erros em todo o sistema.
//
// Descrição: Este script centraliza a lógica para capturar, registrar e responder a erros
//            que ocorrem durante a execução do Apps Script. Ele ajuda a manter a aplicação
//            estável e a fornecer feedback útil ao usuário ou desenvolvedor.
//
// Integrações:
// - Logger.gs: Utiliza o serviço de log para registrar os erros.
// - HtmlService.gs: Pode ser usado para exibir mensagens de erro amigáveis ao usuário no frontend.
// - Todos os Services: Devem utilizar as funções deste script para tratar exceções.
//
// Funções Principais:
// - `handleError(error, functionName)`: Captura e processa um erro, registrando-o e retornando uma resposta padrão.
// - `getErrorMessage(error)`: Extrai uma mensagem de erro legível de um objeto de erro.
// - `showUserError(message)`: Monta uma mensagem de erro amigável ao usuário final.
//
// Observações: É crucial para a robustez da aplicação, garantindo que falhas sejam tratadas
//              de forma controlada e informada.

function getErrorMessage(error) {
  try {
    if (error === null || error === undefined) return 'Erro desconhecido.';
    if (typeof error === 'string') return error;
    if (error.message) return String(error.message);
    return String(error);
  } catch (error) {
    Logger.log("Erro em getErrorMessage: " + error.message);
    throw error;
  }
}

function handleError(error, functionName) {
  var message = getErrorMessage(error);
  var context = functionName || 'desconhecida';
  try {
    if (typeof logError === 'function') logError('[' + context + '] ' + message, error);
    else console.error('[' + context + '] ' + message);
  } catch (e) {
    console.error('[' + context + '] ' + message);
  }
  if (typeof standardReturnFail === 'function') return standardReturnFail(message);
  return { success: false, data: null, error: message };
}

function showUserError(message) {
  return {
    success: false,
    error: message || 'Ocorreu um erro. Tente novamente.',
    userMessage: message || 'Ocorreu um erro. Tente novamente.'
  };
}
