// StudentDataProcessor.gs
//
// Funcionalidade Principal: Processa e normaliza dados brutos de alunos e suas interações.
//
// Descrição: Este script é responsável por limpar, validar e transformar os dados de entrada
//            relacionados aos alunos e suas atividades. Garante que os dados estejam em um
//            formato consistente antes de serem armazenados ou utilizados por outros serviços.
//
// Integrações:
// - AlunoService.gs: Utiliza para persistir os dados processados.
// - ValidationUtils.gs: Para validar a integridade dos dados.
// - UtilityFunctions.gs: Para geração de IDs e normalização.
//
// Funções Principais:
// - `processNewStudentData(rawData)`: Limpa e valida dados de um novo aluno.
// - `normalizeStudentName(name)`: Normaliza o formato do nome do aluno.
// - `formatStudentId(id)`: Formata o ID do aluno para consistência.
//
// Observações: A qualidade dos dados é fundamental para a precisão das análises e relatórios.

function normalizeStudentName(name) {
  try {
    if (!name) return '';
    return String(name)
      .replace(/\s+/g, ' ')
      .trim()
      .toLowerCase()
      .split(' ')
      .map(function (part) { return part ? part.charAt(0).toUpperCase() + part.slice(1) : ''; })
      .join(' ');
  } catch (error) {
    Logger.log("Erro em normalizeStudentName: " + error.message);
    throw error;
  }
}

function formatStudentId(id) {
  try {
    var raw = String(id == null ? '' : id).replace(/[^A-Za-z0-9]/g, '').toUpperCase();
    if (!raw) return '';
    return /^ALU/.test(raw) ? raw : ('ALU-' + raw);
  } catch (error) {
    Logger.log("Erro em formatStudentId: " + error.message);
    throw error;
  }
}

function processNewStudentData(rawData) {
  try {
    rawData = rawData || {};
    var errors = [];
    var nome = normalizeStudentName(rawData.nome || rawData.Nome || rawData.name);
    if (typeof isNotNullOrEmpty === 'function' ? !isNotNullOrEmpty(nome) : !nome) {
      errors.push('Nome obrigatorio.');
    }
    var email = rawData.email || rawData.Email || '';
    if (email && typeof isValidEmail === 'function' && !isValidEmail(email)) {
      errors.push('Email invalido.');
    }
    if (errors.length) return { success: false, errors: errors };
    var processed = {
      ID: rawData.id ? formatStudentId(rawData.id) : (typeof generateUniqueId === 'function' ? generateUniqueId() : String(Date.now())),
      Nome: nome,
      TurmaID: rawData.turmaId || rawData.TurmaID || '',
      Email: email,
      Status: 'ativo'
    };
    return { success: true, data: processed };
  } catch (error) {
    Logger.log("Erro em processNewStudentData: " + error.message);
    throw error;
  }
}
