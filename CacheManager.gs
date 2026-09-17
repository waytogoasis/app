// CacheManager.gs
//
// Funcionalidade Principal: Gerencia o cache de dados para melhorar o desempenho do sistema.
//
// Descrição: Este script fornece funções para armazenar e recuperar dados temporariamente
//            no serviço de cache do Apps Script. Isso ajuda a reduzir o número de chamadas
//            repetitivas à Google Planilha e a APIs externas, melhorando a velocidade e
//            a eficiência da aplicação.
//
// Integrações:
// - CacheService (Apps Script): Serviço nativo para gerenciamento de cache.
// - Diversos Services: Podem utilizar este serviço para armazenar dados frequentemente acessados.
//
// Funções Principais:
// - `putInCache(key, value, expirationInSeconds)`: Armazena um valor (serializado em JSON) no cache.
// - `getFromCache(key)`: Recupera e desserializa um valor do cache (null se ausente/expirado).
// - `removeFromCache(key)`: Remove um valor do cache.
// - `clearAllCache()`: Limpa as chaves rastreadas pelo gerenciador.
//
// Observações: Valores são serializados em JSON; TTL limitado a 6h (limite do CacheService).

var CACHE_INDEX_KEY = '__cachemgr_keys__';

function cacheMgr_() { return CacheService.getScriptCache(); }

function putInCache(key, value, expirationInSeconds) {
  try {
    var ttl = Math.min(Math.max(1, expirationInSeconds || 600), 21600);
    cacheMgr_().put(key, JSON.stringify({ v: value }), ttl);
    var idx = getFromCache(CACHE_INDEX_KEY) || [];
    if (idx.indexOf(key) === -1) { idx.push(key); cacheMgr_().put(CACHE_INDEX_KEY, JSON.stringify({ v: idx }), 21600); }
    return true;
  } catch (e) { return false; }
}

function getFromCache(key) {
  try {
    var raw = cacheMgr_().get(key);
    if (raw === null || raw === undefined) return null;
    var parsed = JSON.parse(raw);
    return parsed && Object.prototype.hasOwnProperty.call(parsed, 'v') ? parsed.v : null;
  } catch (e) { return null; }
}

function removeFromCache(key) {
  try { cacheMgr_().remove(key); return true; } catch (e) { return false; }
}

function clearAllCache() {
  try {
    try {
      var idx = getFromCache(CACHE_INDEX_KEY) || [];
      if (idx.length) cacheMgr_().removeAll(idx);
      cacheMgr_().remove(CACHE_INDEX_KEY);
      return idx.length;
    } catch (e) { return 0; }
  } catch (error) {
    Logger.log("Erro em clearAllCache: " + error.message);
    throw error;
  }
}
