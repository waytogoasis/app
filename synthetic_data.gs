/**
 * Dados sintéticos — Way To Go As Is
 * Gerado em 2026-06-21 01:10:44 por generate_synthetic_data_all_projects.py
 *
 * Execute populateSyntheticData() PELO EDITOR do Apps Script para popular
 * as abas de domínio com ~30 registros cada (valida os gráficos do notebook).
 * Idempotente: limpa as linhas de dados antes de reinserir.
 *
 * NÃO define onOpen() — para não colidir com o menu real do projeto.
 */

function populateSyntheticData() {
  try {
    try {
      try {
        var ss = SpreadsheetApp.getActiveSpreadsheet();
        var results = [];

        // Turmas
        try {
          var sheet_Turmas = ss.getSheetByName('Turmas') || ss.insertSheet('Turmas');
          if (sheet_Turmas.getLastRow() > 1) {
            sheet_Turmas.deleteRows(2, sheet_Turmas.getLastRow() - 1);
          }
          var h_sheet_Turmas = ["ID", "Nome", "Serie", "TotalAlunos", "Professor", "Periodo", "Status"];
          sheet_Turmas.getRange(1, 1, 1, h_sheet_Turmas.length).setValues([h_sheet_Turmas]);
          var d_sheet_Turmas = [
            ["TUR-0001", "Bruno Santos", "1A", 1, "PROF-14", "manhã", "inativo"],
            ["TUR-0002", "Ana Silva", "3C", 16, "PROF-10", "integral", "ativo"],
            ["TUR-0003", "Diego Souza", "1A", 10, "PROF-24", "manhã", "inativo"],
            ["TUR-0004", "Gabriela Rocha", "4A", 15, "PROF-27", "manhã", "ativo"],
            ["TUR-0005", "Felipe Costa", "4A", 22, "PROF-82", "integral", "ativo"],
            ["TUR-0006", "Carla Oliveira", "1A", 11, "PROF-85", "integral", "ativo"],
            ["TUR-0007", "Carla Oliveira", "4A", 3, "PROF-39", "tarde", "inativo"],
            ["TUR-0008", "Carla Oliveira", "1A", 45, "PROF-28", "integral", "ativo"],
            ["TUR-0009", "Felipe Costa", "5B", 12, "PROF-19", "noite", "inativo"],
            ["TUR-0010", "Henrique Alves", "1A", 45, "PROF-86", "tarde", "ativo"],
            ["TUR-0011", "Henrique Alves", "1A", 6, "PROF-54", "tarde", "ativo"],
            ["TUR-0012", "Ana Silva", "2B", 13, "PROF-65", "tarde", "ativo"],
            ["TUR-0013", "Carla Oliveira", "1A", 23, "PROF-77", "manhã", "inativo"],
            ["TUR-0014", "Ana Silva", "3C", 37, "PROF-22", "integral", "ativo"],
            ["TUR-0015", "Bruno Santos", "5B", 28, "PROF-37", "noite", "ativo"],
            ["TUR-0016", "Bruno Santos", "5B", 21, "PROF-82", "integral", "inativo"],
            ["TUR-0017", "Carla Oliveira", "1A", 25, "PROF-81", "manhã", "ativo"],
            ["TUR-0018", "Felipe Costa", "2B", 31, "PROF-78", "tarde", "ativo"],
            ["TUR-0019", "Gabriela Rocha", "5B", 10, "PROF-75", "integral", "ativo"],
            ["TUR-0020", "Eduarda Lima", "3C", 20, "PROF-16", "manhã", "ativo"],
            ["TUR-0021", "Carla Oliveira", "4A", 11, "PROF-10", "manhã", "ativo"],
            ["TUR-0022", "Diego Souza", "5B", 19, "PROF-78", "tarde", "ativo"],
            ["TUR-0023", "Carla Oliveira", "2B", 4, "PROF-42", "manhã", "ativo"],
            ["TUR-0024", "Eduarda Lima", "4A", 18, "PROF-93", "tarde", "ativo"],
            ["TUR-0025", "Bruno Santos", "1A", 27, "PROF-56", "tarde", "ativo"],
            ["TUR-0026", "Carla Oliveira", "1A", 9, "PROF-14", "tarde", "ativo"],
            ["TUR-0027", "Eduarda Lima", "2B", 15, "PROF-26", "tarde", "inativo"],
            ["TUR-0028", "Henrique Alves", "3C", 9, "PROF-99", "manhã", "ativo"],
            ["TUR-0029", "Felipe Costa", "2B", 3, "PROF-26", "manhã", "inativo"],
            ["TUR-0030", "Felipe Costa", "1A", 27, "PROF-42", "integral", "ativo"]
          ];
          sheet_Turmas.getRange(2, 1, d_sheet_Turmas.length, h_sheet_Turmas.length).setValues(d_sheet_Turmas);
          results.push('OK Turmas: ' + d_sheet_Turmas.length + ' registros');
        } catch (e) {
          results.push('ERRO Turmas: ' + e.message);
        }

        // Questionarios
        try {
          var sheet_Questionarios = ss.getSheetByName('Questionarios') || ss.insertSheet('Questionarios');
          if (sheet_Questionarios.getLastRow() > 1) {
            sheet_Questionarios.deleteRows(2, sheet_Questionarios.getLastRow() - 1);
          }
          var h_sheet_Questionarios = ["ID", "QuestionnaireTitle", "Tipo", "Perguntas", "Aplicacoes", "Status", "CreatedAt"];
          sheet_Questionarios.getRange(1, 1, 1, h_sheet_Questionarios.length).setValues([h_sheet_Questionarios]);
          var d_sheet_Questionarios = [
            ["QUE-0001", "Introdução", "tipo_c", 3, 23, "ativo", "2026-04-25 01:10:44"],
            ["QUE-0002", "Avaliação", "tipo_b", 8, 26, "ativo", "2026-04-03 01:10:44"],
            ["QUE-0003", "Revisão", "tipo_a", 5, 5, "ativo", "2026-03-29 01:10:44"],
            ["QUE-0004", "Conceitos", "tipo_c", 4, 27, "ativo", "2026-04-21 01:10:44"],
            ["QUE-0005", "Prática", "tipo_b", 12, 3, "ativo", "2026-04-07 01:10:44"],
            ["QUE-0006", "Revisão", "tipo_a", 4, 2, "ativo", "2026-05-20 01:10:44"],
            ["QUE-0007", "Revisão", "tipo_b", 4, 40, "inativo", "2026-05-01 01:10:44"],
            ["QUE-0008", "Conceitos", "tipo_a", 12, 6, "ativo", "2026-06-05 01:10:44"],
            ["QUE-0009", "Avaliação", "tipo_b", 3, 33, "ativo", "2026-05-05 01:10:44"],
            ["QUE-0010", "Avaliação", "tipo_c", 8, 27, "inativo", "2026-04-23 01:10:44"],
            ["QUE-0011", "Introdução", "tipo_b", 6, 26, "ativo", "2026-06-10 01:10:44"],
            ["QUE-0012", "Revisão", "tipo_c", 9, 24, "ativo", "2026-06-13 01:10:44"],
            ["QUE-0013", "Prática", "tipo_c", 7, 17, "ativo", "2026-03-29 01:10:44"],
            ["QUE-0014", "Introdução", "tipo_b", 9, 26, "ativo", "2026-03-25 01:10:44"],
            ["QUE-0015", "Prática", "tipo_c", 6, 15, "inativo", "2026-05-07 01:10:44"],
            ["QUE-0016", "Introdução", "tipo_b", 12, 25, "ativo", "2026-04-23 01:10:44"],
            ["QUE-0017", "Conceitos", "tipo_a", 5, 25, "ativo", "2026-04-26 01:10:44"],
            ["QUE-0018", "Introdução", "tipo_a", 7, 10, "ativo", "2026-05-24 01:10:44"],
            ["QUE-0019", "Avaliação", "tipo_c", 5, 22, "ativo", "2026-05-17 01:10:44"],
            ["QUE-0020", "Conceitos", "tipo_a", 11, 33, "ativo", "2026-04-13 01:10:44"],
            ["QUE-0021", "Avaliação", "tipo_c", 12, 7, "inativo", "2026-06-02 01:10:44"],
            ["QUE-0022", "Conceitos", "tipo_c", 7, 6, "inativo", "2026-05-18 01:10:44"],
            ["QUE-0023", "Conceitos", "tipo_a", 11, 22, "ativo", "2026-06-03 01:10:44"],
            ["QUE-0024", "Prática", "tipo_b", 5, 24, "ativo", "2026-05-05 01:10:44"],
            ["QUE-0025", "Revisão", "tipo_a", 6, 16, "ativo", "2026-05-26 01:10:44"],
            ["QUE-0026", "Introdução", "tipo_c", 6, 5, "ativo", "2026-05-05 01:10:44"],
            ["QUE-0027", "Revisão", "tipo_b", 9, 2, "ativo", "2026-05-17 01:10:44"],
            ["QUE-0028", "Prática", "tipo_b", 9, 37, "ativo", "2026-04-18 01:10:44"],
            ["QUE-0029", "Prática", "tipo_b", 9, 38, "ativo", "2026-03-31 01:10:44"],
            ["QUE-0030", "Prática", "tipo_a", 8, 16, "inativo", "2026-05-04 01:10:44"]
          ];
          sheet_Questionarios.getRange(2, 1, d_sheet_Questionarios.length, h_sheet_Questionarios.length).setValues(d_sheet_Questionarios);
          results.push('OK Questionarios: ' + d_sheet_Questionarios.length + ' registros');
        } catch (e) {
          results.push('ERRO Questionarios: ' + e.message);
        }

        // Rubricas
        try {
          var sheet_Rubricas = ss.getSheetByName('Rubricas') || ss.insertSheet('Rubricas');
          if (sheet_Rubricas.getLastRow() > 1) {
            sheet_Rubricas.deleteRows(2, sheet_Rubricas.getLastRow() - 1);
          }
          var h_sheet_Rubricas = ["ID", "RubricName", "RubricDescription", "Criterios", "PontuacaoMax", "Status"];
          sheet_Rubricas.getRange(1, 1, 1, h_sheet_Rubricas.length).setValues([h_sheet_Rubricas]);
          var d_sheet_Rubricas = [
            ["RUB-0001", "Henrique Alves", "Registro de sessão experimental", 9, 6.9, "inativo"],
            ["RUB-0002", "Bruno Santos", "Acompanhamento de evolução", 6, 9.5, "ativo"],
            ["RUB-0003", "Bruno Santos", "Dados coletados durante atividade", 8, 6.4, "ativo"],
            ["RUB-0004", "Ana Silva", "Registro de sessão experimental", 12, 9.6, "ativo"],
            ["RUB-0005", "Bruno Santos", "Registro de sessão experimental", 5, 8.8, "ativo"],
            ["RUB-0006", "Felipe Costa", "Registro de sessão experimental", 12, 6.9, "ativo"],
            ["RUB-0007", "Henrique Alves", "Dados coletados durante atividade", 4, 9.3, "ativo"],
            ["RUB-0008", "Ana Silva", "Acompanhamento de evolução", 7, 5.7, "ativo"],
            ["RUB-0009", "Bruno Santos", "Registro de sessão experimental", 5, 6.9, "ativo"],
            ["RUB-0010", "Ana Silva", "Registro de sessão experimental", 11, 7.9, "ativo"],
            ["RUB-0011", "Carla Oliveira", "Acompanhamento de evolução", 4, 6.5, "ativo"],
            ["RUB-0012", "Eduarda Lima", "Dados coletados durante atividade", 3, 7.3, "inativo"],
            ["RUB-0013", "Diego Souza", "Acompanhamento de evolução", 6, 5.5, "ativo"],
            ["RUB-0014", "Ana Silva", "Observação inicial do processo", 10, 9.7, "ativo"],
            ["RUB-0015", "Carla Oliveira", "Dados coletados durante atividade", 4, 5.4, "inativo"],
            ["RUB-0016", "Bruno Santos", "Registro de sessão experimental", 8, 5.8, "ativo"],
            ["RUB-0017", "Eduarda Lima", "Dados coletados durante atividade", 9, 9.4, "inativo"],
            ["RUB-0018", "Eduarda Lima", "Registro de sessão experimental", 10, 5.2, "inativo"],
            ["RUB-0019", "Diego Souza", "Dados coletados durante atividade", 3, 8.5, "inativo"],
            ["RUB-0020", "Felipe Costa", "Dados coletados durante atividade", 6, 9.5, "inativo"],
            ["RUB-0021", "Henrique Alves", "Dados coletados durante atividade", 12, 7.1, "ativo"],
            ["RUB-0022", "Gabriela Rocha", "Dados coletados durante atividade", 6, 7.3, "ativo"],
            ["RUB-0023", "Eduarda Lima", "Observação inicial do processo", 3, 9.6, "ativo"],
            ["RUB-0024", "Carla Oliveira", "Acompanhamento de evolução", 11, 5.3, "ativo"],
            ["RUB-0025", "Felipe Costa", "Observação inicial do processo", 3, 5.1, "ativo"],
            ["RUB-0026", "Henrique Alves", "Dados coletados durante atividade", 5, 5.3, "ativo"],
            ["RUB-0027", "Felipe Costa", "Acompanhamento de evolução", 6, 5.6, "ativo"],
            ["RUB-0028", "Eduarda Lima", "Dados coletados durante atividade", 12, 8.5, "ativo"],
            ["RUB-0029", "Diego Souza", "Acompanhamento de evolução", 11, 9.0, "ativo"],
            ["RUB-0030", "Henrique Alves", "Dados coletados durante atividade", 3, 6.0, "ativo"]
          ];
          sheet_Rubricas.getRange(2, 1, d_sheet_Rubricas.length, h_sheet_Rubricas.length).setValues(d_sheet_Rubricas);
          results.push('OK Rubricas: ' + d_sheet_Rubricas.length + ' registros');
        } catch (e) {
          results.push('ERRO Rubricas: ' + e.message);
        }

        // Pontuacoes
        try {
          var sheet_Pontuacoes = ss.getSheetByName('Pontuacoes') || ss.insertSheet('Pontuacoes');
          if (sheet_Pontuacoes.getLastRow() > 1) {
            sheet_Pontuacoes.deleteRows(2, sheet_Pontuacoes.getLastRow() - 1);
          }
          var h_sheet_Pontuacoes = ["ID", "Data", "Turma", "Pontos", "Nivel", "Avaliador"];
          sheet_Pontuacoes.getRange(1, 1, 1, h_sheet_Pontuacoes.length).setValues([h_sheet_Pontuacoes]);
          var d_sheet_Pontuacoes = [
            ["PON-0001", "2026-05-28 01:10:44", "5B", 364, "medio", "Carla Oliveira"],
            ["PON-0002", "2026-05-19 01:10:44", "4A", 190, "alto", "Ana Silva"],
            ["PON-0003", "2026-05-14 01:10:44", "2B", 599, "alto", "Henrique Alves"],
            ["PON-0004", "2026-05-02 01:10:44", "2B", 362, "baixo", "Felipe Costa"],
            ["PON-0005", "2026-05-13 01:10:44", "3C", 857, "alto", "Diego Souza"],
            ["PON-0006", "2026-04-24 01:10:44", "2B", 161, "baixo", "Carla Oliveira"],
            ["PON-0007", "2026-05-12 01:10:44", "4A", 242, "alto", "Diego Souza"],
            ["PON-0008", "2026-04-25 01:10:44", "5B", 627, "alto", "Carla Oliveira"],
            ["PON-0009", "2026-06-21 01:10:44", "3C", 382, "medio", "Diego Souza"],
            ["PON-0010", "2026-05-02 01:10:44", "3C", 240, "baixo", "Diego Souza"],
            ["PON-0011", "2026-05-25 01:10:44", "4A", 99, "alto", "Felipe Costa"],
            ["PON-0012", "2026-06-19 01:10:44", "4A", 416, "baixo", "Ana Silva"],
            ["PON-0013", "2026-05-17 01:10:44", "4A", 56, "alto", "Carla Oliveira"],
            ["PON-0014", "2026-05-28 01:10:44", "5B", 519, "baixo", "Carla Oliveira"],
            ["PON-0015", "2026-04-29 01:10:44", "2B", 400, "baixo", "Bruno Santos"],
            ["PON-0016", "2026-06-21 01:10:44", "4A", 137, "baixo", "Gabriela Rocha"],
            ["PON-0017", "2026-06-15 01:10:44", "3C", 670, "alto", "Diego Souza"],
            ["PON-0018", "2026-04-25 01:10:44", "5B", 704, "alto", "Carla Oliveira"],
            ["PON-0019", "2026-05-07 01:10:44", "1A", 996, "medio", "Gabriela Rocha"],
            ["PON-0020", "2026-04-23 01:10:44", "3C", 556, "medio", "Gabriela Rocha"],
            ["PON-0021", "2026-05-22 01:10:44", "2B", 342, "medio", "Eduarda Lima"],
            ["PON-0022", "2026-06-13 01:10:44", "2B", 224, "baixo", "Carla Oliveira"],
            ["PON-0023", "2026-06-12 01:10:44", "2B", 951, "alto", "Ana Silva"],
            ["PON-0024", "2026-05-12 01:10:44", "1A", 782, "alto", "Eduarda Lima"],
            ["PON-0025", "2026-06-05 01:10:44", "2B", 364, "alto", "Carla Oliveira"],
            ["PON-0026", "2026-05-07 01:10:44", "2B", 192, "alto", "Felipe Costa"],
            ["PON-0027", "2026-05-27 01:10:44", "1A", 396, "baixo", "Felipe Costa"],
            ["PON-0028", "2026-06-08 01:10:44", "3C", 681, "baixo", "Gabriela Rocha"],
            ["PON-0029", "2026-06-06 01:10:44", "4A", 895, "baixo", "Bruno Santos"],
            ["PON-0030", "2026-04-28 01:10:44", "1A", 554, "baixo", "Eduarda Lima"]
          ];
          sheet_Pontuacoes.getRange(2, 1, d_sheet_Pontuacoes.length, h_sheet_Pontuacoes.length).setValues(d_sheet_Pontuacoes);
          results.push('OK Pontuacoes: ' + d_sheet_Pontuacoes.length + ' registros');
        } catch (e) {
          results.push('ERRO Pontuacoes: ' + e.message);
        }

        // Simulacoes
        try {
          var sheet_Simulacoes = ss.getSheetByName('Simulacoes') || ss.insertSheet('Simulacoes');
          if (sheet_Simulacoes.getLastRow() > 1) {
            sheet_Simulacoes.deleteRows(2, sheet_Simulacoes.getLastRow() - 1);
          }
          var h_sheet_Simulacoes = ["ID", "Data", "Turma", "Tipo", "Media", "Participantes", "Status"];
          sheet_Simulacoes.getRange(1, 1, 1, h_sheet_Simulacoes.length).setValues([h_sheet_Simulacoes]);
          var d_sheet_Simulacoes = [
            ["SIM-0001", "2026-05-27 01:10:44", "4A", "tipo_b", 13, 16, "ativo"],
            ["SIM-0002", "2026-05-21 01:10:44", "3C", "tipo_a", 71, 20, "ativo"],
            ["SIM-0003", "2026-05-11 01:10:44", "5B", "tipo_b", 19, 1, "ativo"],
            ["SIM-0004", "2026-06-07 01:10:44", "4A", "tipo_a", 51, 4, "ativo"],
            ["SIM-0005", "2026-05-10 01:10:44", "4A", "tipo_a", 73, 31, "ativo"],
            ["SIM-0006", "2026-05-28 01:10:44", "1A", "tipo_b", 93, 18, "inativo"],
            ["SIM-0007", "2026-06-19 01:10:44", "3C", "tipo_b", 32, 37, "ativo"],
            ["SIM-0008", "2026-05-01 01:10:44", "2B", "tipo_c", 84, 5, "ativo"],
            ["SIM-0009", "2026-06-16 01:10:44", "4A", "tipo_c", 34, 5, "ativo"],
            ["SIM-0010", "2026-05-16 01:10:44", "3C", "tipo_b", 54, 18, "ativo"],
            ["SIM-0011", "2026-05-16 01:10:44", "1A", "tipo_c", 92, 25, "ativo"],
            ["SIM-0012", "2026-05-15 01:10:44", "3C", "tipo_b", 70, 32, "ativo"],
            ["SIM-0013", "2026-05-01 01:10:44", "2B", "tipo_a", 61, 1, "ativo"],
            ["SIM-0014", "2026-06-20 01:10:44", "3C", "tipo_a", 55, 11, "ativo"],
            ["SIM-0015", "2026-06-17 01:10:44", "5B", "tipo_c", 63, 7, "ativo"],
            ["SIM-0016", "2026-05-06 01:10:44", "1A", "tipo_a", 39, 5, "ativo"],
            ["SIM-0017", "2026-05-16 01:10:44", "2B", "tipo_c", 55, 40, "inativo"],
            ["SIM-0018", "2026-05-23 01:10:44", "5B", "tipo_c", 82, 30, "inativo"],
            ["SIM-0019", "2026-06-19 01:10:44", "5B", "tipo_c", 60, 14, "ativo"],
            ["SIM-0020", "2026-06-11 01:10:44", "4A", "tipo_b", 61, 15, "ativo"],
            ["SIM-0021", "2026-05-01 01:10:44", "5B", "tipo_a", 16, 29, "inativo"],
            ["SIM-0022", "2026-05-09 01:10:44", "1A", "tipo_b", 43, 37, "inativo"],
            ["SIM-0023", "2026-05-11 01:10:44", "5B", "tipo_b", 45, 16, "ativo"],
            ["SIM-0024", "2026-04-22 01:10:44", "5B", "tipo_c", 88, 20, "ativo"],
            ["SIM-0025", "2026-06-14 01:10:44", "5B", "tipo_a", 80, 25, "ativo"],
            ["SIM-0026", "2026-06-02 01:10:44", "2B", "tipo_a", 65, 38, "ativo"],
            ["SIM-0027", "2026-06-10 01:10:44", "2B", "tipo_a", 18, 3, "ativo"],
            ["SIM-0028", "2026-06-14 01:10:44", "5B", "tipo_a", 45, 2, "ativo"],
            ["SIM-0029", "2026-05-21 01:10:44", "4A", "tipo_b", 36, 37, "inativo"],
            ["SIM-0030", "2026-05-01 01:10:44", "3C", "tipo_b", 0, 4, "ativo"]
          ];
          sheet_Simulacoes.getRange(2, 1, d_sheet_Simulacoes.length, h_sheet_Simulacoes.length).setValues(d_sheet_Simulacoes);
          results.push('OK Simulacoes: ' + d_sheet_Simulacoes.length + ' registros');
        } catch (e) {
          results.push('ERRO Simulacoes: ' + e.message);
        }

        Logger.log(results.join('\n'));
        return results;
      } catch (error) {
        Logger.log("Erro em populateSyntheticData: " + error.message);
        throw error; // Re-lança para tratamento superior
      }
    } catch (error) {
      Logger.log("Erro em populateSyntheticData: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em populateSyntheticData: " + error.message);
    throw error;
  }
}
