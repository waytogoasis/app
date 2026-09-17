// StudentAchievementManager.gs
//
// Funcionalidade Principal: Gerencia o reconhecimento e registro de conquistas dos alunos.
//
// Descrição: Este script permite definir critérios para conquistas (ex: "Mestre do Semáforo")
//            e registrar quando os alunos as alcançam. Usado para gamificação e motivação.
//
// Integrações:
// - Google Planilha (abas `ConquistasDef` e `ConquistasAlunos`): definições e concessões.
// - UserService.gs (wtg* helpers): camada de CRUD reutilizável.
// - PontuacaoService.gs: Para verificar critérios de conquista (média de pontuações).
//
// Funções Principais:
// - `defineAchievement(achievementId, criteria, description)`: Define uma nova conquista.
// - `checkAndAwardAchievement(alunoId)`: Verifica critérios e concede conquistas ainda não obtidas.
// - `getAchievementsByAluno(alunoId)`: Retorna as conquistas concedidas a um aluno.

var CONQUISTAS_DEF_SHEET = 'ConquistasDef';
var CONQUISTAS_DEF_HEADERS = ['ID', 'Criteria', 'Descricao', 'CriadoEm', 'AtualizadoEm'];
var CONQUISTAS_ALUNOS_SHEET = 'ConquistasAlunos';
var CONQUISTAS_ALUNOS_HEADERS = ['ID', 'AlunoID', 'ConquistaID', 'ConcedidaEm', 'CriadoEm', 'AtualizadoEm'];

function defineAchievement(achievementId, criteria, description) {
  try {
    return wtgCreateRecord_(CONQUISTAS_DEF_SHEET, CONQUISTAS_DEF_HEADERS, {
      ID: achievementId,
      Criteria: typeof criteria === 'object' ? JSON.stringify(criteria) : (criteria || '{}'),
      Descricao: description || ''
    }, { required: ['ID'] });
  } catch (error) {
    Logger.log("Erro em defineAchievement: " + error.message);
    throw error;
  }
}

function sam_hasAchievement_(alunoId, conquistaId) {
  return wtgReadObjects_(CONQUISTAS_ALUNOS_SHEET).some(function (r) {
    return String(r.AlunoID) === String(alunoId) && String(r.ConquistaID) === String(conquistaId);
  });
}

function sam_alunoMedia_(alunoId) {
  var pts = (typeof getPontuacoesByAluno === 'function') ? getPontuacoesByAluno(alunoId) : [];
  if (!pts.length) return 0;
  return pts.reduce(function (s, p) { return s + (Number(p.Total) || 0); }, 0) / pts.length;
}

function checkAndAwardAchievement(alunoId) {
  try {
    try {
      var defs = wtgReadObjects_(CONQUISTAS_DEF_SHEET);
      var media = sam_alunoMedia_(alunoId);
      var awarded = [];
      defs.forEach(function (def) {
        if (sam_hasAchievement_(alunoId, def.ID)) return;
        var crit; try { crit = JSON.parse(def.Criteria || '{}'); } catch (e) { crit = {}; }
        var met = (crit.minMedia !== undefined && media >= Number(crit.minMedia));
        if (met) {
          wtgCreateRecord_(CONQUISTAS_ALUNOS_SHEET, CONQUISTAS_ALUNOS_HEADERS, {
            AlunoID: alunoId, ConquistaID: def.ID, ConcedidaEm: new Date().toISOString()
          }, { required: ['AlunoID'] });
          awarded.push(def.ID);
        }
      });
      return { success: true, data: { awarded: awarded, media: Math.round(media * 100) / 100 } };
    } catch (error) {
      Logger.log("Erro em checkAndAwardAchievement: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em checkAndAwardAchievement: " + error.message);
    throw error;
  }
}

function getAchievementsByAluno(alunoId) {
  return wtgReadObjects_(CONQUISTAS_ALUNOS_SHEET)
    .filter(function (r) { return String(r.AlunoID || r.alunoid || '') === String(alunoId); });
}
