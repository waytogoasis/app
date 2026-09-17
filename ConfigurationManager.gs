// ConfigurationManager.gs
//
// Funcionalidade Principal: Gerencia as configurações do sistema de forma centralizada.
//
// Descrição: Interface unificada para acessar e modificar as configurações do sistema,
//            delegando ao ConfigService (aba Settings + Script Properties).
//
// Integrações:
// - ConfigService.gs: funções básicas de gerenciamento de configurações.
//
// Funções Principais:
// - `getSystemConfig(key)`: Retorna o valor de uma configuração (com fallback ao default).
// - `setSystemConfig(key, value)`: Define ou atualiza uma configuração do sistema.
// - `loadDefaultConfig()`: Garante a presença das configurações padrão do sistema.

var SYSTEM_DEFAULT_CONFIG = {
  app_name: 'Way To Go As Is',
  idioma_padrao: 'pt-BR',
  tema_padrao: 'claro',
  max_alunos_por_turma: '35',
  meta_media_aprovacao: '70'
};

function getSystemConfig(key) {
  var value = (typeof getSetting === 'function') ? getSetting(key) : null;
  if (value === null || value === undefined) {
    return SYSTEM_DEFAULT_CONFIG[key] !== undefined ? SYSTEM_DEFAULT_CONFIG[key] : null;
  }
  return value;
}

function setSystemConfig(key, value) {
  if (typeof setSetting !== 'function') return { success: false, message: 'ConfigService indisponivel.' };
  var res = setSetting(key, value, 'system');
  return { success: true, data: { key: key, value: value }, raw: res };
}

function loadDefaultConfig() {
  try {
    var applied = [];
    Object.keys(SYSTEM_DEFAULT_CONFIG).forEach(function (key) {
      var current = (typeof getSetting === 'function') ? getSetting(key) : null;
      if (current === null || current === undefined || current === '') {
        setSystemConfig(key, SYSTEM_DEFAULT_CONFIG[key]);
        applied.push(key);
      }
    });
    return { success: true, data: { aplicadas: applied, total: Object.keys(SYSTEM_DEFAULT_CONFIG).length } };
  } catch (error) {
    Logger.log("Erro em loadDefaultConfig: " + error.message);
    throw error;
  }
}
