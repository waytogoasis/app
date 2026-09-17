// StudentManagementUI.gs
//
// Funcionalidade Principal: Funções de Apps Script para interagir com a interface de gerenciamento de alunos.
//
// Descrição: Este script atua como uma ponte entre o frontend HTML de gerenciamento de alunos
//            e o backend `AlunoService.gs`. Ele recebe requisições da UI, chama as funções
//            apropriadas do `AlunoService.gs` e retorna os resultados para a interface.
//
// Integrações:
// - AlunoService.gs: Para realizar operações CRUD de alunos.
// - HtmlService.gs: Para servir a página `StudentManagement.html`.
// - PermissionService.gs: Para verificar permissões antes de executar ações.
//
// Funções Principais:
// - `getStudentsForUI()`: Retorna uma lista de alunos para exibição na UI.
// - `saveStudentFromUI(studentData)`: Salva (cria ou atualiza) um aluno a partir dos dados da UI.
// - `deleteStudentFromUI(studentId)`: Deleta um aluno a partir da UI.
//
// Observações: Garante que as interações da interface do usuário com o backend sejam seguras e eficientes.

/**
 * Retorna lista de alunos para exibição na UI.
 * Inclui validação de permissões e formatação de dados.
 * @param {Object} [filters] - Filtros opcionais (status, turmaId, etc.)
 * @return {Object} Lista de alunos com metadata
 */
function getStudentsForUI(filters) {
  try {
    // Verifica permissão
    var currentUser = studentManagementCurrentUser_();
    if (typeof can === 'function' && !can(currentUser, 'alunos.view')) {
      return {
        success: false,
        error: 'Permissão negada para visualizar alunos'
      };
    }

    filters = filters || {};
    var alunos = [];

    // Obtém todos os alunos
    try {
      if (typeof getAllAlunos === 'function') {
        alunos = getAllAlunos();
      } else if (typeof wtgReadObjects_ === 'function') {
        alunos = wtgReadObjects_('Alunos').filter(function(a) {
          return (a.Status || a.status || 'ativo').toLowerCase() !== 'inativo';
        });
      } else {
        throw new Error('Serviço de alunos não disponível');
      }
    } catch (e) {
      Logger.log("Erro ao obter alunos: " + e.message);
      return { success: false, error: 'Erro ao buscar alunos: ' + e.message };
    }

    // Aplica filtros
    if (filters.status) {
      alunos = alunos.filter(function(a) {
        return (a.Status || a.status || '').toLowerCase() === filters.status.toLowerCase();
      });
    }

    if (filters.turmaId) {
      alunos = alunos.filter(function(a) {
        return String(a.TurmaID || a.turmaId || '') === String(filters.turmaId);
      });
    }

    if (filters.search) {
      var searchTerm = filters.search.toLowerCase();
      alunos = alunos.filter(function(a) {
        var nome = (a.Nome || a.nome || '').toLowerCase();
        var id = String(a.ID || a.id || '');
        return nome.indexOf(searchTerm) !== -1 || id.indexOf(searchTerm) !== -1;
      });
    }

    // Enriquece dados com informações adicionais
    var enrichedAlunos = alunos.map(function(aluno) {
      var enriched = {
        id: aluno.ID || aluno.id,
        nome: aluno.Nome || aluno.nome || '',
        turmaId: aluno.TurmaID || aluno.turmaId || '',
        status: aluno.Status || aluno.status || 'ativo',
        criadoEm: aluno.CriadoEm || aluno.criadoEm || '',
        atualizadoEm: aluno.AtualizadoEm || aluno.atualizadoEm || ''
      };

      // Adiciona estatísticas
      try {
        if (typeof getPontuacoesByAluno === 'function') {
          var pontuacoes = getPontuacoesByAluno(enriched.id);
          enriched.totalSimulacoes = pontuacoes.length;
          
          if (pontuacoes.length > 0) {
            var soma = pontuacoes.reduce(function(sum, p) {
              return sum + (Number(p.Total || p.total) || 0);
            }, 0);
            enriched.pontuacaoMedia = Math.round(soma / pontuacoes.length * 100) / 100;
          } else {
            enriched.pontuacaoMedia = 0;
          }
        }
      } catch (e) {
        enriched.totalSimulacoes = 0;
        enriched.pontuacaoMedia = 0;
      }

      return enriched;
    });

    return {
      success: true,
      data: enrichedAlunos,
      count: enrichedAlunos.length,
      timestamp: new Date().toISOString()
    };
  } catch (error) {
    Logger.log("Erro em getStudentsForUI: " + error.message);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Salva (cria ou atualiza) um aluno a partir dos dados da UI.
 * @param {Object} studentData - Dados do aluno (com ou sem ID para criar/atualizar)
 * @return {Object} Resultado da operação
 */
function saveStudentFromUI(studentData) {
  try {
    // Verifica permissão
    var currentUser = studentManagementCurrentUser_();
    var isUpdate = !!(studentData.id || studentData.ID);
    var action = isUpdate ? 'alunos.edit' : 'alunos.create';
    
    if (typeof can === 'function' && !can(currentUser, action)) {
      return {
        success: false,
        error: 'Permissão negada para ' + (isUpdate ? 'editar' : 'criar') + ' alunos'
      };
    }

    if (!studentData) {
      return { success: false, error: 'Dados do aluno não fornecidos' };
    }

    // Validação básica
    if (!studentData.nome && !studentData.Nome) {
      return { success: false, error: 'Nome do aluno é obrigatório' };
    }

    var result;

    if (isUpdate) {
      // Atualização de aluno existente
      var alunoId = studentData.id || studentData.ID;
      var updateData = {
        Nome: studentData.nome || studentData.Nome,
        TurmaID: studentData.turmaId || studentData.TurmaID || '',
        Status: studentData.status || studentData.Status || 'ativo'
      };

      if (typeof updateAluno === 'function') {
        result = updateAluno(alunoId, updateData);
      } else if (typeof wtgUpdateRecordById_ === 'function') {
        result = wtgUpdateRecordById_('Alunos', alunoId, updateData);
      } else {
        throw new Error('Serviço de atualização de aluno não disponível');
      }

      // Registra auditoria
      try {
        if (typeof logAudit === 'function') {
          logAudit(
            currentUser.id || currentUser.ID || 'user',
            'UPDATE',
            'Alunos',
            alunoId,
            { updated: updateData }
          );
        }
      } catch (auditError) {
        Logger.log("Aviso: não foi possível registrar auditoria: " + auditError.message);
      }

    } else {
      // Criação de novo aluno
      var createData = {
        Nome: studentData.nome || studentData.Nome,
        TurmaID: studentData.turmaId || studentData.TurmaID || '',
        Status: 'ativo'
      };

      if (typeof createAluno === 'function') {
        result = createAluno(createData);
      } else {
        throw new Error('Serviço de criação de aluno não disponível');
      }

      // Registra auditoria
      try {
        if (typeof logAudit === 'function' && result.success) {
          logAudit(
            currentUser.id || currentUser.ID || 'user',
            'CREATE',
            'Alunos',
            result.data ? (result.data.ID || result.data.id) : 'new',
            { created: createData }
          );
        }
      } catch (auditError) {
        Logger.log("Aviso: não foi possível registrar auditoria: " + auditError.message);
      }
    }

    return result;
  } catch (error) {
    Logger.log("Erro em saveStudentFromUI: " + error.message);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Deleta (inativa) um aluno a partir da UI.
 * @param {string|number} studentId - ID do aluno a ser deletado
 * @return {Object} Resultado da operação
 */
function deleteStudentFromUI(studentId) {
  try {
    // Verifica permissão
    var currentUser = studentManagementCurrentUser_();
    if (typeof can === 'function' && !can(currentUser, 'alunos.delete')) {
      return {
        success: false,
        error: 'Permissão negada para deletar alunos'
      };
    }

    if (!studentId) {
      return { success: false, error: 'ID do aluno não fornecido' };
    }

    // Verifica se o aluno existe
    var aluno = null;
    try {
      if (typeof getAlunoById === 'function') {
        var alunoResult = getAlunoById(studentId);
        if (alunoResult && alunoResult.success) {
          aluno = alunoResult.data;
        }
      }
    } catch (e) {
      Logger.log("Aviso: não foi possível verificar existência do aluno: " + e.message);
    }

    if (!aluno) {
      return { success: false, error: 'Aluno não encontrado: ' + studentId };
    }

    // Soft delete (inativação)
    var result;
    if (typeof deleteAluno === 'function') {
      result = deleteAluno(studentId);
    } else if (typeof wtgUpdateRecordById_ === 'function') {
      result = wtgUpdateRecordById_('Alunos', studentId, { 
        Status: 'inativo', 
        Ativo: false,
        InativadoEm: new Date().toISOString()
      });
    } else {
      throw new Error('Serviço de deleção de aluno não disponível');
    }

    // Registra auditoria
    try {
      if (typeof logAudit === 'function' && result.success) {
        logAudit(
          currentUser.id || currentUser.ID || 'user',
          'DELETE',
          'Alunos',
          studentId,
          { 
            alunoNome: aluno.Nome || aluno.nome,
            action: 'soft_delete'
          }
        );
      }
    } catch (auditError) {
      Logger.log("Aviso: não foi possível registrar auditoria: " + auditError.message);
    }

    return result;
  } catch (error) {
    Logger.log("Erro em deleteStudentFromUI: " + error.message);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Função auxiliar para obter usuário atual
 */
function studentManagementCurrentUser_() {
  try {
    if (typeof getCurrentSessionUser === 'function') {
      return getCurrentSessionUser();
    } else if (typeof getSession === 'function') {
      var session = getSession();
      if (session && session.userId && typeof getUserById === 'function') {
        return getUserById(session.userId);
      }
    }
    return { id: 'unknown', role: 'user' };
  } catch (e) {
    return { id: 'unknown', role: 'user' };
  }
}
