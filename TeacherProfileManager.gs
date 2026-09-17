// TeacherProfileManager.gs
//
// Funcionalidade Principal: Gerencia perfis detalhados dos professores.
//
// Descrição: Armazena e recupera informações adicionais sobre os professores
//            (especializações, histórico de turmas, observações) que complementam o UserService.
//
// Integrações:
// - Google Planilha (aba `PerfisProfessores`): Armazenamento dos perfis.
// - UserService.gs (wtg* helpers): camada de CRUD reutilizável.
//
// Funções Principais:
// - `createTeacherProfile(teacherId, profileData)`: Cria um perfil detalhado para um professor.
// - `getTeacherProfile(teacherId)`: Retorna o perfil de um professor (Dados parseado).
// - `updateTeacherProfile(teacherId, newProfileData)`: Atualiza (merge) o perfil de um professor.

var PERFIS_PROF_SHEET = 'PerfisProfessores';
var PERFIS_PROF_HEADERS = ['ID', 'ProfessorID', 'Dados', 'CriadoEm', 'AtualizadoEm'];

function tpm_findRaw_(teacherId) {
  try {
    return wtgReadObjects_(PERFIS_PROF_SHEET)
      .filter(function (p) { return String(p.ProfessorID || p.professorid || '') === String(teacherId); })[0] || null;
  } catch (error) {
    Logger.log("Erro em tpm_findRaw_: " + error.message);
    throw error;
  }
}

function createTeacherProfile(teacherId, profileData) {
  try {
    if (String(teacherId || '').trim() === '') return { success: false, message: 'teacherId obrigatorio.' };
    if (tpm_findRaw_(teacherId)) return { success: false, message: 'Perfil ja existe para este professor.' };
    return wtgCreateRecord_(PERFIS_PROF_SHEET, PERFIS_PROF_HEADERS, {
      ProfessorID: teacherId, Dados: JSON.stringify(profileData || {})
    }, { required: ['ProfessorID'] });
  } catch (error) {
    Logger.log("Erro em createTeacherProfile: " + error.message);
    throw error;
  }
}

function getTeacherProfile(teacherId) {
  try {
    var raw = tpm_findRaw_(teacherId);
    if (!raw) return { success: false, message: 'Perfil nao encontrado.' };
    var dados; try { dados = JSON.parse(raw.Dados || '{}'); } catch (e) { dados = {}; }
    return { success: true, data: { ID: raw.ID, ProfessorID: raw.ProfessorID, dados: dados } };
  } catch (error) {
    Logger.log("Erro em getTeacherProfile: " + error.message);
    throw error;
  }
}

function updateTeacherProfile(teacherId, newProfileData) {
  try {
    try {
      var raw = tpm_findRaw_(teacherId);
      if (!raw) return createTeacherProfile(teacherId, newProfileData);
      var atual; try { atual = JSON.parse(raw.Dados || '{}'); } catch (e) { atual = {}; }
      Object.keys(newProfileData || {}).forEach(function (k) { atual[k] = newProfileData[k]; });
      return wtgUpdateRecordById_(PERFIS_PROF_SHEET, raw.ID, { Dados: JSON.stringify(atual) });
    } catch (error) {
      Logger.log("Erro em updateTeacherProfile: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em updateTeacherProfile: " + error.message);
    throw error;
  }
}
