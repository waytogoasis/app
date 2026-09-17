// ApiResponse.gs
//
// Funcionalidade Principal: Envelope de resposta padronizado entre backend e frontend.
//
// Descrição: Define o CONTRATO único que todas as funções server-side chamadas via
//            google.script.run devem retornar. Padroniza sucesso e erro para que o
//            frontend trate respostas de forma consistente (ver ClientRouter.html →
//            callServer). Baseado no padrão "envelope" da skill api-patterns/response.
//
// Formato:
//   sucesso  -> { success: true,  data: <qualquer>, message: '', error: null }
//   erro     -> { success: false, data: null, message: '<humano>', error: { code } }
//   (login/logout adicionam `redirectUrl` ao envelope de sucesso)
//
// Integrações:
// - AuthUI.gs, *UI.gs, ApiEndpoints.gs: usam apiOk()/apiFail() como retorno.
// - ClientRouter.html (callServer): interpreta o campo `success`.

/** Envelope de sucesso. */
function apiOk(data, message) {
  return {
    success: true,
    data: (data === undefined ? null : data),
    message: message || '',
    error: null
  };
}

/** Envelope de erro. `code` é uma etiqueta estável (VALIDATION, AUTH, etc.). */
function apiFail(message, code) {
  return {
    success: false,
    data: null,
    message: message || 'Ocorreu um erro.',
    error: { code: code || 'ERROR' }
  };
}
