// QuestionnaireManager.gs
//
// Funcionalidade Principal: Gerencia a criação e aplicação de questionários para alunos e professores.
//
// Descrição: Permite criar questionários personalizados, registrar respostas e agregar resultados.
//
// Integrações:
// - Google Planilha (abas `Questionarios` e `Respostas`): Armazenamento.
// - UserService.gs (wtg* helpers): camada de CRUD reutilizável.
//
// Funções Principais:
// - `createQuestionnaire(title, questions)`: Cria um novo questionário.
// - `submitResponse(questionnaireId, userId, answers)`: Registra as respostas de um usuário.
// - `getQuestionnaireResults(questionnaireId)`: Retorna respostas e agregação por pergunta.

var QUESTIONARIOS_SHEET = 'Questionarios';
var QUESTIONARIOS_HEADERS = ['ID', 'Titulo', 'Perguntas', 'CriadoEm', 'AtualizadoEm'];
var RESPOSTAS_SHEET = 'Respostas';
var RESPOSTAS_HEADERS = ['ID', 'QuestionarioID', 'UserID', 'Respostas', 'CriadoEm', 'AtualizadoEm'];

function createQuestionnaire(title, questions) {
  try {
    if (String(title || '').trim() === '') return { success: false, message: 'Titulo obrigatorio.' };
    return wtgCreateRecord_(QUESTIONARIOS_SHEET, QUESTIONARIOS_HEADERS, {
      Titulo: title, Perguntas: JSON.stringify(questions || [])
    }, { required: ['Titulo'] });
  } catch (error) {
    Logger.log("Erro em createQuestionnaire: " + error.message);
    throw error;
  }
}

function submitResponse(questionnaireId, userId, answers) {
  try {
    if (String(questionnaireId || '').trim() === '') return { success: false, message: 'questionnaireId obrigatorio.' };
    return wtgCreateRecord_(RESPOSTAS_SHEET, RESPOSTAS_HEADERS, {
      QuestionarioID: questionnaireId, UserID: userId || '', Respostas: JSON.stringify(answers || {})
    }, { required: ['QuestionarioID'] });
  } catch (error) {
    Logger.log("Erro em submitResponse: " + error.message);
    throw error;
  }
}

function getQuestionnaireResults(questionnaireId) {
  try {
    var respostas = wtgReadObjects_(RESPOSTAS_SHEET)
      .filter(function (r) { return String(r.QuestionarioID || r.questionarioid || '') === String(questionnaireId); })
      .map(function (r) { try { return JSON.parse(r.Respostas || '{}'); } catch (e) { return {}; } });
    // Agrega por pergunta: distribuição de respostas.
    var agregacao = {};
    respostas.forEach(function (resp) {
      Object.keys(resp).forEach(function (pergunta) {
        var valor = String(resp[pergunta]);
        agregacao[pergunta] = agregacao[pergunta] || {};
        agregacao[pergunta][valor] = (agregacao[pergunta][valor] || 0) + 1;
      });
    });
    return { questionnaireId: questionnaireId, totalRespostas: respostas.length, agregacao: agregacao };
  } catch (error) {
    Logger.log("Erro em getQuestionnaireResults: " + error.message);
    throw error;
  }
}
