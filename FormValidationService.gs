// FormValidationService.gs
//
// Funcionalidade Principal: Valida dados de formulários antes do processamento no backend.
//
// Descrição: Funções para validar conjuntos de campos de formulários (login, aluno, pontuação),
//            reutilizando ValidationUtils e retornando lista de erros amigável.
//
// Integrações:
// - ValidationUtils.gs: funções de validação genéricas.
// - Interfaces HTML: chamadas para validação antes do envio.
//
// Funções Principais:
// - `validateLoginForm(username, password)`: Valida os campos do formulário de login.
// - `validateAlunoForm(alunoData)`: Valida os campos do formulário de aluno.
// - `validatePontuacaoForm(pontuacaoData)`: Valida os campos do formulário de pontuação.

function fvs_result_(errors) {
  return { valid: errors.length === 0, errors: errors };
}

function validateLoginForm(username, password) {
  try {
    var errors = [];
    if (typeof isNotNullOrEmpty === 'function' ? !isNotNullOrEmpty(username) : !username) errors.push('Usuário obrigatório.');
    if (typeof isNotNullOrEmpty === 'function' ? !isNotNullOrEmpty(password) : !password) errors.push('Senha obrigatória.');
    return fvs_result_(errors);
  } catch (error) {
    Logger.log("Erro em validateLoginForm: " + error.message);
    throw error;
  }
}

function validateAlunoForm(alunoData) {
  try {
    alunoData = alunoData || {};
    var errors = [];
    var nome = alunoData.Nome || alunoData.nome;
    if (typeof isNotNullOrEmpty === 'function' ? !isNotNullOrEmpty(nome) : !nome) errors.push('Nome obrigatório.');
    var email = alunoData.Email || alunoData.email;
    if (email && typeof isValidEmail === 'function' && !isValidEmail(email)) errors.push('E-mail inválido.');
    return fvs_result_(errors);
  } catch (error) {
    Logger.log("Erro em validateAlunoForm: " + error.message);
    throw error;
  }
}

function validatePontuacaoForm(pontuacaoData) {
  try {
    pontuacaoData = pontuacaoData || {};
    var errors = [];
    if (typeof isNotNullOrEmpty === 'function' ? !isNotNullOrEmpty(pontuacaoData.alunoId || pontuacaoData.AlunoID) : !(pontuacaoData.alunoId || pontuacaoData.AlunoID)) {
      errors.push('Aluno obrigatório.');
    }
    var indicadores = pontuacaoData.indicadores || pontuacaoData.pontuacoes || {};
    Object.keys(indicadores).forEach(function (k) {
      var ok = (typeof isNumeric === 'function') ? isNumeric(indicadores[k]) : !isNaN(Number(indicadores[k]));
      if (!ok) errors.push('Indicador não numérico: ' + k);
      else if (Number(indicadores[k]) < 0 || Number(indicadores[k]) > 100) errors.push('Indicador fora de 0-100: ' + k);
    });
    return fvs_result_(errors);
  } catch (error) {
    Logger.log("Erro em validatePontuacaoForm: " + error.message);
    throw error;
  }
}
