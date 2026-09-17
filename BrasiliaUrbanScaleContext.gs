// BrasiliaUrbanScaleContext.gs
//
// Funcionalidade Principal: Centraliza o contexto urbano de Brasília que dá identidade ao
//                           projeto "Brincando no Trânsito Seguro".
//
// Descrição: Codifica as escalas urbanas do Plano Piloto — com ênfase na escala gregária e
//            na escala residencial das superquadras — e as traduz em conteúdo pedagógico,
//            cenários de simulação e notas de psicologia do trânsito. É a fonte única de
//            verdade sobre o que torna o trânsito de Brasília específico: a vida gregária
//            preservada dentro de cada superquadra, o uso das entrequadras para o comércio
//            de vizinhança e a previsão de que, para alcançar a superquadra pretendida, o
//            cidadão muitas vezes precisa atravessar (a pé ou de carro) a superquadra vizinha.
//            Esse "atravessar o quintal do vizinho" é o coração da convivência cidadã que o
//            projeto quer ensinar.
//
// Integrações:
// - SimulationEnvironmentManager.gs: expõe o layout do cenário "superquadra" para a simulação.
// - WeeklyDynamics.gs / PedagogicalContentManager.gs: alimenta atividades de cidadania.
// - LegalFrameworkEval.gs: ancora "o maior cuida do menor" e a prioridade do pedestre na
//   realidade espacial das quadras.
//
// Funções Principais:
// - `getBrasiliaScales()`: Retorna as escalas urbanas e o que cada uma ensina sobre trânsito.
// - `getBrasiliaTrafficRules()`: Regras de trânsito contextualizadas para Brasília.
// - `getSuperquadraModel()`: Retorna o modelo espacial de uma superquadra (vias, entrequadra,
//   acesso pela quadra vizinha).
// - `getTrafficPsychologyNotes()`: Notas de psicologia do trânsito situadas em Brasília.
// - `getNeighboringSuperquadraScenario()`: O cenário-chave de cidadania (atravessar a quadra
//   vizinha para chegar à pretendida).
// - `getSuperquadraSimulationLayout()`: Layout pronto para o ambiente de simulação.
// - `getCitizenshipActivities(weekNumber)`: Atividades de cidadania por semana, situadas nas escalas.

/**
 * Escalas urbanas de Brasília (Lúcio Costa) relevantes para a educação no trânsito.
 * A escala gregária é a do encontro e do convívio; a residencial é a da superquadra,
 * onde se preserva o caráter gregário de bairro.
 */
const BRASILIA_SCALES = {
  monumental: {
    nome: 'Escala Monumental',
    descricao: 'Eixo dos poderes e dos grandes espaços cívicos; trânsito de fluxo e de representação.',
    licaoTransito: 'Distâncias longas e travessias amplas exigem planejamento e atenção sustentada.'
  },
  residencial: {
    nome: 'Escala Residencial',
    descricao: 'A superquadra: blocos sobre pilotis, áreas verdes e vias internas de baixa velocidade.',
    licaoTransito: 'O pedestre (criança, idoso) tem prioridade; o carro é hóspede dentro da quadra.'
  },
  gregaria: {
    nome: 'Escala Gregária',
    descricao: 'O espaço do encontro e do comércio local; cada superquadra preserva sua vida de bairro.',
    licaoTransito: 'Conviver é negociar o espaço: ao cruzar a quadra vizinha, respeita-se quem ali mora.'
  },
  bucolica: {
    nome: 'Escala Bucólica',
    descricao: 'Os vazios verdes que separam e respiram entre as quadras.',
    licaoTransito: 'Áreas de transição entre quadras pedem redução de velocidade e olhar atento.'
  }
};

/**
 * Regras de trânsito trabalhadas como leitura da cidade de Brasília, não apenas
 * como memorização. A ideia é que o aluno reconheça a regra no lugar concreto:
 * faixa da entrequadra, via interna da superquadra, eixinho, comércio local,
 * ponto de ônibus e áreas de convivência.
 */
var BRASILIA_TRAFFIC_RULES = [
  {
    id: 'faixa_pedestre_df',
    regra: 'Respeitar a faixa de pedestre e aguardar a travessia completa.',
    contextoBrasilia: 'Brasília transformou o respeito à faixa em símbolo de cidadania; nas entrequadras, a faixa organiza o encontro entre morador, estudante, ciclista e motorista.',
    reconhecer: 'Faixa pintada, pedestre aguardando, fluxo de veículos aproximando e necessidade de contato visual.',
    comportamentoEsperado: 'Parar antes da faixa, sinalizar intenção de atravessar como pedestre e só seguir quando a passagem estiver segura.'
  },
  {
    id: 'velocidade_via_interna',
    regra: 'Reduzir a velocidade em vias internas, áreas escolares, residenciais e de convivência.',
    contextoBrasilia: 'Nas superquadras, pilotis, jardins e vias internas aproximam crianças, idosos, ciclistas e carros em um mesmo espaço cotidiano.',
    reconhecer: 'Via estreita, blocos residenciais, áreas verdes, crianças brincando, ausência de semáforo e placas de regulamentação.',
    comportamentoEsperado: 'Entrar devagar, observar laterais e pilotis, evitar buzina e nunca usar a quadra como atalho de velocidade.'
  },
  {
    id: 'preferencia_pedestre_vulneravel',
    regra: 'Dar prioridade ao pedestre e proteger usuários mais vulneráveis.',
    contextoBrasilia: 'A escala residencial faz o carro circular dentro de uma área que primeiro é de moradia e convivência.',
    reconhecer: 'Criança, idoso, pessoa com deficiência, travessia longa, mochila escolar ou baixa visibilidade entre veículos estacionados.',
    comportamentoEsperado: 'Ceder passagem, esperar sem pressionar e manter distância de segurança.'
  },
  {
    id: 'sinalizacao_horizontal_vertical',
    regra: 'Reconhecer e obedecer sinalização horizontal, vertical e semafórica.',
    contextoBrasilia: 'Eixos, eixinhos, tesourinhas, retornos e entrequadras exigem leitura rápida de placas, faixas, setas e semáforos.',
    reconhecer: 'Placas de PARE, Dê a Preferência, velocidade máxima, semáforo, faixa, linha de retenção e sentido de circulação.',
    comportamentoEsperado: 'Nomear a sinalização, explicar a regra com as próprias palavras e escolher a ação segura antes de avançar.'
  },
  {
    id: 'convivencia_quadra_vizinha',
    regra: 'Ao atravessar a quadra vizinha, agir como visitante em espaço de convivência.',
    contextoBrasilia: 'A organização das superquadras prevê deslocamentos por quadras vizinhas e entrequadras comerciais compartilhadas.',
    reconhecer: 'Entrada em via interna de outra quadra, comércio local compartilhado, pedestres chegando aos blocos e áreas verdes sem barreira física.',
    comportamentoEsperado: 'Reduzir, dar preferência, não cortar área verde e respeitar o ritmo de quem mora ou circula ali.'
  }
];

/**
 * Modelo espacial de uma superquadra e a previsão de acesso pela quadra vizinha.
 * Numeração tipo "SQS 308 / 108": quadras pares e ímpares ladeiam o eixo; o acesso
 * viário a uma quadra frequentemente passa pela via interna ou entrequadra da vizinha.
 */
var SUPERQUADRA_MODEL = {
  exemplo: 'SQN 108 (pretendida) acessada a partir da SQN 109 (vizinha)',
  caracteristicas: [
    'Blocos residenciais sobre pilotis, térreo livre e permeável.',
    'Faixa verde arborizada circundando a quadra (limite gregário, não muro).',
    'Via interna única de baixa velocidade, sem travessias semaforizadas.',
    'Entrequadra comercial compartilhada entre quadras vizinhas (ex.: 108/109).',
    'Sinalização de faixa, preferência, velocidade e parada precisa ser lida no contexto do lugar.'
  ],
  acessoVizinho: {
    descricao: 'Para chegar à superquadra pretendida, o trajeto costuma atravessar a via ' +
               'interna ou a entrequadra da superquadra vizinha. O motorista e o pedestre ' +
               'entram, por alguns instantes, no espaço de convívio de outra comunidade.',
    dever: 'Comportar-se como visitante: reduzir, dar preferência a quem mora e brinca ali, ' +
           'não usar a quadra vizinha como atalho de velocidade.',
    licaoCidadania: 'A cidade é gregária por projeto: o espaço do vizinho é também espaço comum, ' +
                    'e a travessia exige cortesia, não pressa.',
    licaoRegras: 'A regra de trânsito ganha sentido quando o aluno reconhece onde está: faixa da ' +
                 'entrequadra, via interna residencial, área verde, pilotis e acesso à quadra vizinha.'
  }
};

/**
 * Notas de psicologia do trânsito situadas na realidade de Brasília.
 */
var TRAFFIC_PSYCHOLOGY_NOTES = [
  {
    tema: 'Percepção de risco em vias internas',
    nota: 'A baixa velocidade e a ausência de semáforos na superquadra reduzem a percepção ' +
          'de risco; trabalha-se a antecipação (criança que surge entre os pilotis).'
  },
  {
    tema: 'Pertencimento e território',
    nota: 'Atravessar a quadra vizinha ativa a noção de território; o objetivo é converter ' +
          'a defesa de "minha quadra" em hospitalidade e respeito mútuo.'
  },
  {
    tema: 'Atenção dividida e impulsividade',
    nota: 'Entrequadras misturam pedestres, ciclistas e carros; exercita-se controle ' +
          'inibitório (parar antes de cruzar) e atenção dividida.'
  },
  {
    tema: 'Empatia e a regra "o maior cuida do menor"',
    nota: 'A escala gregária dá rosto a quem está na via; a empatia substitui a regra ' +
          'abstrata pela relação concreta com o vizinho.'
  },
  {
    tema: 'Reconhecimento situado das regras',
    nota: 'O aluno aprende a nomear a regra e a justificar sua aplicação no lugar concreto: ' +
          'faixa da entrequadra, PARE na via interna, preferência no acesso à quadra vizinha ' +
          'e velocidade compatível com área residencial.'
  }
];

/** Retorna as escalas urbanas e suas lições de trânsito. */
function getBrasiliaScales() {
  try {
    return JSON.parse(JSON.stringify(BRASILIA_SCALES));
  } catch (error) {
    Logger.log("Erro em getBrasiliaScales: " + error.message);
    throw error;
  }
}

/** Retorna regras de trânsito contextualizadas para a realidade de Brasília. */
function getBrasiliaTrafficRules() {
  try {
    return JSON.parse(JSON.stringify(BRASILIA_TRAFFIC_RULES));
  } catch (error) {
    Logger.log("Erro em getBrasiliaTrafficRules: " + error.message);
    throw error;
  }
}

/** Retorna o modelo espacial da superquadra, incluindo o acesso pela quadra vizinha. */
function getSuperquadraModel() {
  try {
    return JSON.parse(JSON.stringify(SUPERQUADRA_MODEL));
  } catch (error) {
    Logger.log("Erro em getSuperquadraModel: " + error.message);
    throw error;
  }
}

/** Retorna as notas de psicologia do trânsito situadas em Brasília. */
function getTrafficPsychologyNotes() {
  try {
    return TRAFFIC_PSYCHOLOGY_NOTES.slice();
  } catch (error) {
    Logger.log("Erro em getTrafficPsychologyNotes: " + error.message);
    throw error;
  }
}

/**
 * Cenário-chave de cidadania: para chegar à superquadra pretendida, atravessa-se a vizinha.
 * Pensado para virar atividade/simulação e item de avaliação do marco legal.
 */
function getNeighboringSuperquadraScenario() {
  return {
    id: 'cidadania_quadra_vizinha',
    titulo: 'Atravessando a superquadra vizinha',
    contexto: SUPERQUADRA_MODEL.acessoVizinho.descricao,
    objetivo: 'Chegar à superquadra pretendida reconhecendo as regras de trânsito e respeitando quem mora na quadra vizinha.',
    deveres: [
      'Reconhecer a sinalização antes de agir: PARE, Dê a Preferência, faixa, velocidade e sentido da via.',
      'Reduzir a velocidade ao entrar na quadra vizinha.',
      'Dar preferência ao pedestre, à criança e ao idoso (o maior cuida do menor).',
      'Usar a faixa quando ela existir e não atravessar no improviso se houver travessia sinalizada próxima.',
      'Não cortar caminho pela área verde nem usar a via interna como atalho rápido.',
      'Agradecer/ceder na entrequadra compartilhada (108/109).'
    ],
    dimensoes: ['MarcoLegal', 'Cidadania', 'Cognitiva'],
    indicadores: [
      'reconheceu_sinalizacao',
      'explicou_regra_no_contexto_brasilia',
      'reduziu_velocidade',
      'deu_preferencia_pedestre',
      'usou_faixa_quando_existente',
      'respeitou_area_convivio'
    ]
  };
}

/**
 * Layout do ambiente de simulação representando uma superquadra e o acesso pela vizinha.
 * Compatível com o formato consumido por SimulationEnvironmentManager.gs.
 */
function getSuperquadraSimulationLayout() {
  return {
    nome: 'Superquadra (SQN 108 via 109)',
    patio: { largura: 24, altura: 18 },
    regrasContextuais: getBrasiliaTrafficRules(),
    elementos: [
      { id: 'entrequadra_108_109', tipo: 'entrequadra', x: 12, y: 2, texto: 'Entrequadra 108/109' },
      { id: 'faixa_entrequadra', tipo: 'faixa_pedestre', x: 12, y: 4, texto: 'Faixa da entrequadra' },
      { id: 'via_interna_109', tipo: 'via_interna', x: 6, y: 9, velocidadeMax: 30, texto: 'Quadra vizinha 109' },
      { id: 'pilotis_109', tipo: 'pilotis', x: 5, y: 12 },
      { id: 'area_verde_109', tipo: 'area_verde', x: 9, y: 6 },
      { id: 'faixa_pedestre_eixinho', tipo: 'faixa_pedestre', x: 12, y: 10 },
      { id: 'via_interna_108', tipo: 'via_interna', x: 18, y: 9, velocidadeMax: 30, texto: 'Quadra pretendida 108' },
      { id: 'pilotis_108', tipo: 'pilotis', x: 19, y: 13 },
      { id: 'crianca_brincando', tipo: 'pedestre_prioritario', x: 8, y: 11, texto: 'Criança nos pilotis' },
      { id: 'idoso_na_faixa', tipo: 'pedestre_prioritario', x: 13, y: 4, texto: 'Idoso na faixa' },
      { id: 'placa_de_preferencia', tipo: 'placa', x: 11, y: 8, texto: 'DÊ A PREFERÊNCIA' },
      { id: 'placa_pare_via_interna', tipo: 'placa', x: 7, y: 8, texto: 'PARE' },
      { id: 'placa_velocidade_30', tipo: 'placa', x: 6, y: 6, texto: '30 km/h' }
    ],
    objetivoCidadao: getNeighboringSuperquadraScenario().objetivo
  };
}

/**
 * Atividades de cidadania por semana, ancoradas nas escalas de Brasília.
 * Complementa o WEEKLY_PLAN com a dimensão "Cidadania".
 */
function getCitizenshipActivities(weekNumber) {
  try {
    var porSemana = {
      1: [
        { id: 'c1_minha_quadra', nome: 'Mapeando a minha superquadra e seus sinais', dimensao: 'Cidadania' },
        { id: 'c1_regras_visiveis', nome: 'Reconhecendo faixa, placa e sentido da via no caminho da escola', dimensao: 'MarcoLegal' }
      ],
      2: [
        { id: 'c2_entrequadra', nome: 'A entrequadra como espaço de encontro e travessia segura', dimensao: 'Cidadania' },
        { id: 'c2_faixa_df', nome: 'Brasília e a cultura de respeito à faixa de pedestre', dimensao: 'MarcoLegal' }
      ],
      3: [
        { id: 'c3_quadra_vizinha', nome: 'Atravessando a superquadra vizinha com respeito', dimensao: 'Cidadania' },
        { id: 'c3_preferencia_contexto', nome: 'Aplicando PARE, Dê a Preferência e velocidade baixa na via interna', dimensao: 'MarcoLegal' }
      ],
      4: [
        { id: 'c4_guardioes_quadra', nome: 'Guardiões da convivência na quadra', dimensao: 'Cidadania' },
        { id: 'c4_auditoria_regras', nome: 'Auditoria cidadã: explicar a regra certa para cada ponto da superquadra', dimensao: 'MarcoLegal' }
      ]
    };
    if (weekNumber === undefined || weekNumber === null) return porSemana;
    return (porSemana[weekNumber] || []).slice();
  } catch (error) {
    Logger.log("Erro em getCitizenshipActivities: " + error.message);
    throw error;
  }
}
