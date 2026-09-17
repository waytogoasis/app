// Semana1Logic.gs
//
// Funcionalidade Principal: Contém a lógica específica e as atividades para a Semana 1 do projeto.
//
// Descrição: "Semiótica e Percepção Visual" — pintura do pátio, debate sobre o significado de
//            sinais e jogos de controle inibitório imediato. Delega ao plano semanal compartilhado
//            (WeeklyDynamics).
//
// Funções Principais:
// - `initSemana1()`: Inicializa as atividades e configurações para a Semana 1.
// - `getSemana1Activities()`: Retorna a lista de atividades planejadas para a semana.
// - `evaluateSemana1Activity(activityId, alunoId, result)`: Registra e avalia o resultado.

function initSemana1() { return wtgInitWeek_(1); }
function getSemana1Activities() { return wtgWeekActivities_(1); }
function evaluateSemana1Activity(activityId, alunoId, result) { return wtgEvaluateWeekActivity_(1, activityId, alunoId, result); }
