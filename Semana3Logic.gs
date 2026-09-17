// Semana3Logic.gs
//
// Funcionalidade Principal: Contém a lógica específica e as atividades para a Semana 3 do projeto.
//
// Descrição: "Prioridades e Tomada de Decisão" — cruzamentos, placa "Dê a Preferência" e
//            situações de atenção dividida e decisão rápida. Delega ao plano semanal compartilhado.
//
// Funções Principais:
// - `initSemana3()`: Inicializa as atividades e configurações para a Semana 3.
// - `getSemana3Activities()`: Retorna a lista de atividades planejadas para a semana.
// - `evaluateSemana3Activity(activityId, alunoId, result)`: Registra e avalia o resultado.

function initSemana3() { return wtgInitWeek_(3); }
function getSemana3Activities() { return wtgWeekActivities_(3); }
function evaluateSemana3Activity(activityId, alunoId, result) { return wtgEvaluateWeekActivity_(3, activityId, alunoId, result); }
