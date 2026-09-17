// SessionService.gs
//
// SUPERSEDED / DESATIVADO (FROTA): este arquivo era uma copia do modelo
// SGTE e definia globais createSession_sgteLegacy/getSession_sgteLegacy/invalidateSession_sgteLegacy/
// getCurrentSessionUser_sgteLegacy que COLIDIAM com o owner nativo SessionManager.gs.
// Como o Apps Script compartilha escopo global e este arquivo carrega depois
// (ordem alfabetica), suas definicoes sobrescreviam as nativas e chamavam
// sanitizeUserForClient_ (inexistente neste projeto), gerando
// "Erro ao processar login: sanitizeUserForClient_ is not defined".
//
// As funcoes foram renomeadas com sufixo _sgteLegacy para nao colidir.
// O owner canonico de sessao e SessionManager.gs (createSession_sgteLegacy(userId, role),
// getSession_sgteLegacy, clearSession, isSessionActive). Nada chama as versoes abaixo.

const CURRENT_SESSION_KEY_sgteLegacy = "SGTE_CURRENT_SESSION";
const SESSION_TTL_SECONDS_sgteLegacy = 21600;

function createSession_sgteLegacy(userId) {
  try {
    const user = getUserById(Number(userId));
    if (!user) {
      throw new Error("Não foi possível criar a sessão: usuário inexistente.");
    }

    const session = {
      userId: user.ID,
      createdAt: new Date().toISOString()
    };

    CacheService.getUserCache().put(
      CURRENT_SESSION_KEY_sgteLegacy,
      JSON.stringify(session),
      SESSION_TTL_SECONDS_sgteLegacy
    );
    return session;
  } catch (error) {
    Logger.log("Erro em createSession_sgteLegacy: " + error.message);
    throw error;
  }
}

function getSession_sgteLegacy() {
  try {
    const serialized = CacheService.getUserCache().get(CURRENT_SESSION_KEY_sgteLegacy);
    if (!serialized) {
      return null;
    }

    try {
      return JSON.parse(serialized);
    } catch (parseError) {
      invalidateSession_sgteLegacy();
      return null;
    }
  } catch (error) {
    Logger.log("Erro em getSession_sgteLegacy: " + error.message);
    throw error;
  }
}

function invalidateSession_sgteLegacy() {
  CacheService.getUserCache().remove(CURRENT_SESSION_KEY_sgteLegacy);
}

function getCurrentSessionUser_sgteLegacy() {
  try {
    const session = getSession_sgteLegacy();
    if (!session || !session.userId) {
      return null;
    }

    const user = getUserById(Number(session.userId));
    if (!user || user.Status === "Inactive") {
      invalidateSession_sgteLegacy();
      return null;
    }
    return user;
  } catch (error) {
    Logger.log("Erro em getCurrentSessionUser_sgteLegacy: " + error.message);
    throw error;
  }
}
