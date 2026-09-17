// Semana4Logic.gs
//
// Funcionalidade Principal: Contém a lógica específica e as atividades para a Semana 4 do projeto.
//
// Descrição: "Consolidação Ética e Avaliação" — criação de cartazes, debate "tamanho e idade
//            definem deveres" e preparação dos alunos como guardiões do trânsito. Delega ao
//            plano semanal compartilhado.
//
// Funções Principais:
// - `initSemana4()`: Inicializa as atividades e configurações para a Semana 4.
// - `getSemana4Activities()`: Retorna a lista de atividades planejadas para a semana.
// - `evaluateSemana4Activity(activityId, alunoId, result)`: Registra e avalia o resultado.

function initSemana4() { return wtgInitWeek_(4); }
function getSemana4Activities() { return wtgWeekActivities_(4); }
function evaluateSemana4Activity(activityId, alunoId, result) { return wtgEvaluateWeekActivity_(4, activityId, alunoId, result); }
