// ValidationUtils.gs
//
// Funcionalidade Principal: Fornece funções utilitárias para validação de dados de entrada.
//
// Descrição: Este script contém funções genéricas para validar diferentes tipos de dados,
//            como e-mails, senhas, IDs, datas, etc. É essencial para garantir a integridade
//            dos dados antes que sejam armazenados na Google Planilha ou processados por
//            outros serviços.
//
// Integrações:
// - Todos os Services (AuthService, UserService, AlunoService, etc.): Utilizam estas funções
//   para validar os dados recebidos do frontend ou de outras fontes.
//
// Funções Principais:
// - `isValidEmail(email)`: Valida o formato de um endereço de e-mail.
// - `isValidPassword(password)`: Valida a complexidade de uma senha (comprimento mínimo).
// - `isNumeric(value)`: Verifica se um valor é numérico.
// - `isNotNullOrEmpty(value)`: Verifica se um valor não é nulo ou vazio.
// - `isValidDate(dateString)`: Valida o formato e a validade de uma string de data.
//
// Observações: As regras de validação podem ser personalizadas conforme a necessidade do projeto.

function isValidEmail(email) {
  return typeof email === 'string' && /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

function isValidPassword(password) {
  // Mínimo 6 caracteres, com ao menos uma letra e um número.
  return typeof password === 'string' && password.length >= 6 &&
    /[A-Za-z]/.test(password) && /[0-9]/.test(password);
}

function isNumeric(value) {
  try {
    if (value === null || value === undefined || value === '') return false;
    return !isNaN(Number(value)) && isFinite(Number(value));
  } catch (error) {
    Logger.log("Erro em isNumeric: " + error.message);
    throw error;
  }
}

function isNotNullOrEmpty(value) {
  try {
    if (value === null || value === undefined) return false;
    if (typeof value === 'string') return value.trim() !== '';
    if (Array.isArray(value)) return value.length > 0;
    return true;
  } catch (error) {
    Logger.log("Erro em isNotNullOrEmpty: " + error.message);
    throw error;
  }
}

function isValidDate(dateString) {
  if (!dateString) return false;
  var d = new Date(dateString);
  return d instanceof Date && !isNaN(d.getTime());
}
