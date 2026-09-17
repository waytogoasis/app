// StudentProfileManager.gs
//
// Funcionalidade Principal: Gerencia perfis detalhados dos alunos.
//
// Descrição: Este script é responsável por armazenar e recuperar informações adicionais
//            sobre os alunos, como histórico escolar, necessidades especiais, ou observações
//            individuais que complementam os dados básicos do AlunoService.gs.
//
// Integrações:
// - Google Planilha (aba `PerfisAlunos`): Armazenamento dos perfis.
// - UserService.gs (wtg* helpers): camada de CRUD reutilizável.
// - AlunoService.gs: Para associar perfis a alunos existentes.
//
// Funções Principais:
// - `createStudentProfile(alunoId, profileData)`: Cria um perfil detalhado para um aluno.
// - `getStudentProfile(alunoId)`: Retorna o perfil de um aluno (Dados em JSON parseado).
// - `updateStudentProfile(alunoId, newProfileData)`: Atualiza (merge) o perfil de um aluno.

var PERFIS_ALUNOS_SHEET = 'PerfisAlunos';
var PERFIS_ALUNOS_HEADERS = ['ID', 'AlunoID', 'Dados', 'CriadoEm', 'AtualizadoEm'];

function spm_findRaw_(alunoId) {
  try {
    return wtgReadObjects_(PERFIS_ALUNOS_SHEET)
      .filter(function (p) { return String(p.AlunoID || p.alunoid || '') === String(alunoId); })[0] || null;
  } catch (error) {
    Logger.log("Erro em spm_findRaw_: " + error.message);
    throw error;
  }
}

function createStudentProfile(alunoId, profileData) {
  try {
    if (String(alunoId || '').trim() === '') return { success: false, message: 'alunoId obrigatorio.' };
    if (spm_findRaw_(alunoId)) return { success: false, message: 'Perfil ja existe para este aluno.' };
    return wtgCreateRecord_(PERFIS_ALUNOS_SHEET, PERFIS_ALUNOS_HEADERS, {
      AlunoID: alunoId, Dados: JSON.stringify(profileData || {})
    }, { required: ['AlunoID'] });
  } catch (error) {
    Logger.log("Erro em createStudentProfile: " + error.message);
    throw error;
  }
}

function getStudentProfile(alunoId) {
  try {
    var raw = spm_findRaw_(alunoId);
    if (!raw) return { success: false, message: 'Perfil nao encontrado.' };
    var dados; try { dados = JSON.parse(raw.Dados || '{}'); } catch (e) { dados = {}; }
    return { success: true, data: { ID: raw.ID, AlunoID: raw.AlunoID, dados: dados } };
  } catch (error) {
    Logger.log("Erro em getStudentProfile: " + error.message);
    throw error;
  }
}

function updateStudentProfile(alunoId, newProfileData) {
  try {
    try {
      var raw = spm_findRaw_(alunoId);
      if (!raw) return createStudentProfile(alunoId, newProfileData);
      var atual; try { atual = JSON.parse(raw.Dados || '{}'); } catch (e) { atual = {}; }
      Object.keys(newProfileData || {}).forEach(function (k) { atual[k] = newProfileData[k]; });
      return wtgUpdateRecordById_(PERFIS_ALUNOS_SHEET, raw.ID, { Dados: JSON.stringify(atual) });
    } catch (error) {
      Logger.log("Erro em updateStudentProfile: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em updateStudentProfile: " + error.message);
    throw error;
  }
}
