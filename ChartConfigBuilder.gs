// ChartConfigBuilder.gs
//
// Funcionalidade Principal: Constrói configurações para gráficos Chart.js.
//
// Descrição: Cria objetos de configuração (type/data/options) compatíveis com Chart.js,
//            abstraindo a montagem para o frontend.
//
// Integrações:
// - ChartGenerator.gs: usa estes builders para gerar as configurações finais.
//
// Funções Principais:
// - `buildLineChartConfig(labels, datasets, options)`: Gráfico de linha.
// - `buildBarChartConfig(labels, datasets, options)`: Gráfico de barras.
// - `buildPieChartConfig(labels, data, options)`: Gráfico de pizza.
// - `mergeChartOptions(defaultOptions, customOptions)`: Mescla (deep) opções.

var CHART_DEFAULT_OPTIONS = { responsive: true, plugins: { legend: { position: 'top' } } };

function mergeChartOptions(defaultOptions, customOptions) {
  try {
    defaultOptions = defaultOptions || {};
    customOptions = customOptions || {};
    var out = {};
    Object.keys(defaultOptions).forEach(function (k) { out[k] = defaultOptions[k]; });
    Object.keys(customOptions).forEach(function (k) {
      if (customOptions[k] && typeof customOptions[k] === 'object' && !Array.isArray(customOptions[k]) &&
          out[k] && typeof out[k] === 'object' && !Array.isArray(out[k])) {
        out[k] = mergeChartOptions(out[k], customOptions[k]);
      } else {
        out[k] = customOptions[k];
      }
    });
    return out;
  } catch (error) {
    Logger.log("Erro em mergeChartOptions: " + error.message);
    throw error;
  }
}

function buildLineChartConfig(labels, datasets, options) {
  return { type: 'line', data: { labels: labels || [], datasets: datasets || [] }, options: mergeChartOptions(CHART_DEFAULT_OPTIONS, options) };
}

function buildBarChartConfig(labels, datasets, options) {
  return { type: 'bar', data: { labels: labels || [], datasets: datasets || [] }, options: mergeChartOptions(CHART_DEFAULT_OPTIONS, options) };
}

function buildPieChartConfig(labels, data, options) {
  return {
    type: 'pie',
    data: { labels: labels || [], datasets: [{ data: data || [] }] },
    options: mergeChartOptions(CHART_DEFAULT_OPTIONS, options)
  };
}
