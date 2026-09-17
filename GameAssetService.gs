/**
 * Catálogo visual remoto do Way To Go As Is.
 *
 * Resolve os sete arquivos canônicos pelo nome exato em FOLDER_ID. Variantes
 * locais de formato não participam do runtime do webapp.
 */
var WAY_TO_GO_ASSET_PROJECT = 'way-to-go-as-is';
var WAY_TO_GO_ASSET_FILES = Object.freeze([
  'avatar_navegador.png',
  'cartas_tradeoffs.svg',
  'comparador_cenarios.webp',
  'encruzilhada_cenarios_hero.webp',
  'mapa_decisoes.svg',
  'marcadores_evidencia.svg',
  'selo_caminho_justificado.svg'
]);

var WAY_TO_GO_ASSET_MIME_TYPES = Object.freeze({
  png: ['image/png'],
  svg: ['image/svg+xml'],
  webp: ['image/webp']
});

function getGameAssetManifest() {
  return WayToGoAssetService_getManifest_();
}

function WayToGoAssetService_getManifest_() {
  var manifest = {
    project: WAY_TO_GO_ASSET_PROJECT,
    ok: false,
    assets: {},
    assetItems: [],
    missing: [],
    duplicates: [],
    invalidMimeTypes: [],
    error: ''
  };

  try {
    var folder = WayToGoAssetService_resolveFolder_();
    WAY_TO_GO_ASSET_FILES.forEach(function(name) {
      var iterator = folder.getFilesByName(name);
      var files = [];
      while (iterator.hasNext()) files.push(iterator.next());

      if (!files.length) {
        manifest.missing.push(name);
        return;
      }
      if (files.length > 1) {
        manifest.duplicates.push(name);
        return;
      }

      var file = files[0];
      var mimeType = file.getMimeType();
      if (!WayToGoAssetService_hasExpectedMimeType_(name, mimeType)) {
        manifest.invalidMimeTypes.push({ name: name, mimeType: mimeType });
        return;
      }

      var item = {
        name: name,
        mimeType: mimeType,
        url: 'https://drive.google.com/uc?export=view&id=' + encodeURIComponent(file.getId())
      };
      manifest.assets[name] = item.url;
      manifest.assetItems.push(item);
    });

    manifest.ok = !manifest.missing.length &&
      !manifest.duplicates.length &&
      !manifest.invalidMimeTypes.length;

    if (!manifest.ok) {
      manifest.assets = {};
      manifest.assetItems = [];
      manifest.error = WayToGoAssetService_buildError_(manifest);
    }
  } catch (error) {
    manifest.assets = {};
    manifest.assetItems = [];
    manifest.error = error && error.message ? error.message : String(error);
  }
  return manifest;
}

function WayToGoAssetService_resolveFolder_() {
  var folderId = String(PropertiesService.getScriptProperties().getProperty('FOLDER_ID') || '').trim();
  if (!folderId) {
    throw new Error('Configure FOLDER_ID nas propriedades do script para carregar os assets do Way To Go As Is.');
  }
  return DriveApp.getFolderById(folderId);
}

function WayToGoAssetService_hasExpectedMimeType_(name, mimeType) {
  var extension = name.split('.').pop().toLowerCase();
  return (WAY_TO_GO_ASSET_MIME_TYPES[extension] || []).indexOf(mimeType) !== -1;
}

function WayToGoAssetService_buildError_(manifest) {
  var parts = [];
  if (manifest.missing.length) parts.push('ausentes: ' + manifest.missing.join(', '));
  if (manifest.duplicates.length) parts.push('duplicados: ' + manifest.duplicates.join(', '));
  if (manifest.invalidMimeTypes.length) {
    parts.push('tipos inválidos: ' + manifest.invalidMimeTypes.map(function(item) {
      return item.name + ' (' + item.mimeType + ')';
    }).join(', '));
  }
  return 'Catálogo visual indisponível — ' + parts.join('; ') + '.';
}
