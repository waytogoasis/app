// ParentCommunicationManager.gs
//
// Funcionalidade Principal: Gerencia a comunicação com os pais ou responsáveis dos alunos.
//
// Descrição: Facilita o envio de relatórios de progresso, comunicados gerais e agendamento de
//            reuniões. O envio de e-mail é best-effort; toda comunicação é registrada na aba
//            `Comunicacoes` para rastreabilidade.
//
// Integrações:
// - EmailService.gs / MailApp: envio de e-mails (best-effort).
// - RelatorioService.gs: relatório de progresso do aluno.
// - UserService.gs (wtg* helpers): persistência das comunicações.
//
// Funções Principais:
// - `sendProgressReportToParent(alunoId)`: Envia o relatório de progresso de um aluno aos pais.
// - `sendGeneralAnnouncementToParents(message)`: Registra/dispara um comunicado geral.
// - `scheduleParentMeeting(alunoId, date, time)`: Agenda (registra) uma reunião com os pais.

var COMUNICACOES_SHEET = 'Comunicacoes';
var COMUNICACOES_HEADERS = ['ID', 'Tipo', 'AlunoID', 'Destinatario', 'Assunto', 'Mensagem', 'Status', 'CriadoEm', 'AtualizadoEm'];

function pcm_emailDoResponsavel_(alunoId) {
  var aluno = (typeof getAlunoById === 'function') ? getAlunoById(alunoId) : null;
  var dados = aluno && aluno.data ? aluno.data : null;
  if (dados && (dados.EmailResponsavel || dados.emailResponsavel)) return dados.EmailResponsavel || dados.emailResponsavel;
  if (typeof getStudentProfile === 'function') {
    var perfil = getStudentProfile(alunoId);
    if (perfil && perfil.success && perfil.data.dados && perfil.data.dados.emailResponsavel) return perfil.data.dados.emailResponsavel;
  }
  return '';
}

function pcm_trySendEmail_(to, subject, body) {
  if (!to) return false;
  try {
    if (typeof sendEmail === 'function') { sendEmail(to, subject, body); return true; }
    if (typeof MailApp !== 'undefined' && MailApp.sendEmail) { MailApp.sendEmail(to, subject, body); return true; }
  } catch (e) {
    if (typeof logError === 'function') logError('Falha ao enviar email para ' + to, e);
  }
  return false;
}

function pcm_registrar_(tipo, alunoId, destinatario, assunto, mensagem, enviado) {
  return wtgCreateRecord_(COMUNICACOES_SHEET, COMUNICACOES_HEADERS, {
    Tipo: tipo, AlunoID: alunoId || '', Destinatario: destinatario || '',
    Assunto: assunto || '', Mensagem: mensagem || '', Status: enviado ? 'enviado' : 'registrado'
  }, { required: ['Tipo'] });
}

function sendProgressReportToParent(alunoId) {
  try {
    if (String(alunoId || '').trim() === '') return { success: false, message: 'alunoId obrigatorio.' };
    var rel = (typeof generateRelatorioAluno === 'function') ? generateRelatorioAluno(alunoId) : null;
    if (rel && rel.success === false) return rel;
    var email = pcm_emailDoResponsavel_(alunoId);
    var media = (rel && rel.data && rel.data.progresso) ? rel.data.progresso.media : 0;
    var assunto = 'Relatório de progresso';
    var mensagem = 'Resumo de progresso do aluno. Média atual: ' + media + '.';
    var enviado = pcm_trySendEmail_(email, assunto, mensagem);
    var reg = pcm_registrar_('relatorio_progresso', alunoId, email, assunto, mensagem, enviado);
    return { success: true, data: { enviado: enviado, destinatario: email, registro: reg.data } };
  } catch (error) {
    Logger.log("Erro em sendProgressReportToParent: " + error.message);
    throw error;
  }
}

function sendGeneralAnnouncementToParents(message) {
  try {
    if (String(message || '').trim() === '') return { success: false, message: 'Mensagem vazia.' };
    var alunos = (typeof getAllAlunos === 'function') ? getAllAlunos() : [];
    var enviados = 0;
    alunos.forEach(function (a) {
      var email = pcm_emailDoResponsavel_(a.ID || a.id);
      if (pcm_trySendEmail_(email, 'Comunicado', message)) enviados++;
    });
    pcm_registrar_('comunicado_geral', '', 'todos', 'Comunicado', message, enviados > 0);
    return { success: true, data: { totalPais: alunos.length, enviados: enviados } };
  } catch (error) {
    Logger.log("Erro em sendGeneralAnnouncementToParents: " + error.message);
    throw error;
  }
}

function scheduleParentMeeting(alunoId, date, time) {
  try {
    if (String(alunoId || '').trim() === '') return { success: false, message: 'alunoId obrigatorio.' };
    var quando = String(date || '') + (time ? (' ' + time) : '');
    var email = pcm_emailDoResponsavel_(alunoId);
    var mensagem = 'Reunião agendada para ' + quando + '.';
    var enviado = pcm_trySendEmail_(email, 'Convite para reunião', mensagem);
    var reg = pcm_registrar_('reuniao', alunoId, email, 'Convite para reunião', mensagem, enviado);
    return { success: true, data: { quando: quando, enviado: enviado, registro: reg.data } };
  } catch (error) {
    Logger.log("Erro em scheduleParentMeeting: " + error.message);
    throw error;
  }
}
