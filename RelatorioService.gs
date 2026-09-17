// RelatorioService.gs
//
// Funcionalidade Principal: Gera relatórios e consultas complexas a partir dos dados do sistema.
//
// Descrição: Este script consolida dados de diferentes abas da Google Planilha para criar
//            relatórios abrangentes sobre o desempenho dos alunos, o histórico de simulações
//            e a eficácia das avaliações.
//
// Integrações:
// - PontuacaoService.gs / SimulacaoService.gs / AlunoService.gs: fontes de dados de domínio.
// - UserService.gs (wtg* helpers): leitura de abas.
//
// Funções Principais:
// - `generateRelatorioGeral()`: Gera um relatório consolidado de todas as atividades.
// - `generateRelatorioAluno(alunoId)`: Gera um relatório detalhado para um aluno específico.
// - `getProgressoAluno(alunoId)`: Calcula e retorna o progresso de um aluno ao longo do tempo.
// - `getEstatisticasGerais()`: Retorna estatísticas agregadas do sistema.

function getEstatisticasGerais() {
  try {
    var alunos = (typeof getAllAlunos === 'function') ? getAllAlunos() : wtgReadObjects_('Alunos');
    var simulacoes = (typeof getAllSimulations === 'function') ? getAllSimulations() : wtgReadObjects_('Simulacoes');
    var pontuacoes = wtgReadObjects_('Pontuacoes');
    var totais = pontuacoes.map(function (p) { return Number(p.Total) || 0; });
    var media = totais.length ? Math.round(totais.reduce(function (a, b) { return a + b; }, 0) / totais.length * 100) / 100 : 0;
    return {
      totalAlunos: alunos.length,
      totalSimulacoes: simulacoes.length,
      simulacoesFinalizadas: simulacoes.filter(function (s) { return String(s.Status) === 'finalizada'; }).length,
      totalPontuacoes: pontuacoes.length,
      mediaGeral: media
    };
  } catch (error) {
    Logger.log("Erro em getEstatisticasGerais: " + error.message);
    throw error;
  }
}

function getProgressoAluno(alunoId) {
  try {
    var pontuacoes = (typeof getPontuacoesByAluno === 'function') ? getPontuacoesByAluno(alunoId)
      : wtgReadObjects_('Pontuacoes').filter(function (p) { return String(p.AlunoID) === String(alunoId); });
    var serie = pontuacoes
      .sort(function (a, b) { return new Date(a.CriadoEm) - new Date(b.CriadoEm); })
      .map(function (p) { return { data: p.CriadoEm, total: Number(p.Total) || 0, simulacaoId: p.SimulacaoID }; });
    var primeira = serie.length ? serie[0].total : 0;
    var ultima = serie.length ? serie[serie.length - 1].total : 0;
    return {
      alunoId: alunoId,
      avaliacoes: serie.length,
      serie: serie,
      evolucao: Math.round((ultima - primeira) * 100) / 100,
      media: serie.length ? Math.round(serie.reduce(function (s, x) { return s + x.total; }, 0) / serie.length * 100) / 100 : 0
    };
  } catch (error) {
    Logger.log("Erro em getProgressoAluno: " + error.message);
    throw error;
  }
}

function generateRelatorioAluno(alunoId) {
  try {
    var aluno = (typeof getAlunoById === 'function') ? getAlunoById(alunoId) : wtgFindRecordById_('Alunos', alunoId);
    var dadosAluno = aluno && aluno.data ? aluno.data : (aluno || null);
    if (!dadosAluno) return { success: false, message: 'Aluno nao encontrado.' };
    var simulacoes = (typeof getSimulationsByAluno === 'function') ? getSimulationsByAluno(alunoId) : [];
    return {
      success: true,
      data: {
        aluno: dadosAluno,
        simulacoes: simulacoes,
        progresso: getProgressoAluno(alunoId),
        geradoEm: new Date().toISOString()
      }
    };
  } catch (error) {
    Logger.log("Erro em generateRelatorioAluno: " + error.message);
    throw error;
  }
}

function generateRelatorioGeral() {
  try {
    var alunos = (typeof getAllAlunos === 'function') ? getAllAlunos() : wtgReadObjects_('Alunos');
    return {
      success: true,
      data: {
        estatisticas: getEstatisticasGerais(),
        porAluno: alunos.map(function (a) {
          var id = a.ID || a.id;
          return { alunoId: id, nome: a.Nome || a.nome, progresso: getProgressoAluno(id) };
        }),
        geradoEm: new Date().toISOString()
      }
    };
  } catch (error) {
    Logger.log("Erro em generateRelatorioGeral: " + error.message);
    throw error;
  }
}
