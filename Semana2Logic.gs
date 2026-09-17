// Semana2Logic.gs
//
// Funcionalidade Principal: Contém a lógica específica e as atividades para a Semana 2 do projeto.
//
// Descrição: "Coordenação e Controle de Velocidade" — veículos-tremzinho, placa "Reduza a
//            Velocidade" e introdução do papel da polícia. Delega ao plano semanal compartilhado.
//
// Funções Principais:
// - `initSemana2()`: Inicializa as atividades e configurações para a Semana 2.
// - `getSemana2Activities()`: Retorna a lista de atividades planejadas para a semana.
// - `evaluateSemana2Activity(activityId, alunoId, result)`: Registra e avalia o resultado.

function initSemana2() { return wtgInitWeek_(2); }
function getSemana2Activities() { return wtgWeekActivities_(2); }
function evaluateSemana2Activity(activityId, alunoId, result) { return wtgEvaluateWeekActivity_(2, activityId, alunoId, result); }
