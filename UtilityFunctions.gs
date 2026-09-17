// UtilityFunctions.gs
//
// Funcionalidade Principal: Contém funções utilitárias diversas que podem ser usadas em todo o sistema.
//
// Descrição: Este script agrupa funções de uso geral que não se encaixam em categorias mais específicas,
//            mas que são úteis para diversas partes da aplicação. Isso inclui manipulação de strings,
//            datas, arrays, e outras operações comuns.
//
// Integrações:
// - Diversos Services: Podem utilizar estas funções para operações comuns.
//
// Funções Principais:
// - `formatDate(date, format)`: Formata uma data para um padrão específico.
// - `capitalizeFirstLetter(string)`: Capitaliza a primeira letra de uma string.
// - `removeDuplicates(array)`: Remove elementos duplicados de um array.
// - `generateUniqueId()`: Gera um ID único para novos registros.
//
// Observações: Ajuda a evitar a duplicação de código e a manter a consistência.

function formatDate(date, format) {
  try {
    if (!date) return '';
    var d = (date instanceof Date) ? date : new Date(date);
    if (isNaN(d.getTime())) return '';
    var tz = (typeof Session !== 'undefined' && Session.getScriptTimeZone) ? Session.getScriptTimeZone() : 'UTC';
    return Utilities.formatDate(d, tz, format || 'dd/MM/yyyy HH:mm');
  } catch (error) {
    Logger.log("Erro em formatDate: " + error.message);
    throw error;
  }
}

function capitalizeFirstLetter(string) {
  try {
    string = String(string == null ? '' : string);
    return string ? string.charAt(0).toUpperCase() + string.slice(1) : '';
  } catch (error) {
    Logger.log("Erro em capitalizeFirstLetter: " + error.message);
    throw error;
  }
}

function removeDuplicates(array) {
  try {
    var seen = [], out = [];
    (array || []).forEach(function(item) {
      var key = (item && typeof item === 'object') ? JSON.stringify(item) : item;
      if (seen.indexOf(key) === -1) { seen.push(key); out.push(item); }
    });
    return out;
  } catch (error) {
    Logger.log("Erro em removeDuplicates: " + error.message);
    throw error;
  }
}

function generateUniqueId() {
  try { return Utilities.getUuid(); }
  catch (e) { return 'id-' + Date.now() + '-' + Math.floor(Math.random() * 1e6); }
}
