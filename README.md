# Way To Go As Is — "Brincando no Trânsito Seguro"

## Objetivo principal

Este é um projeto de **educação gamificada em segurança e cidadania no trânsito**, voltado a crianças. Seu propósito não é avaliar alunos em abstrato, e sim **formar pedestres e futuros condutores conscientes** por meio de simulações lúdicas, dinâmicas semanais e recompensas (gamificação).

Dois eixos dão identidade ao projeto e devem orientar qualquer mudança:

1. **Psicologia do trânsito** — percepção de risco, controle inibitório (pare/siga), atenção dividida, empatia e a regra "o maior cuida do menor". Veja `BrasiliaUrbanScaleContext.gs` (`getTrafficPsychologyNotes`).
2. **Educação para o trânsito como leitura da cidade** — o aluno não apenas memoriza placas; ele reconhece faixa, PARE, Dê a Preferência, velocidade, semáforo e sentido da via em situações concretas de Brasília: escola, pilotis, via interna, entrequadra, eixo/eixinho e quadra vizinha. Veja `getBrasiliaTrafficRules()` e `getLegalFrameworkBrasiliaIndicators()`.
3. **As escalas urbanas de Brasília** — particularmente a **escala gregária/residencial das superquadras**. O cenário-chave de cidadania é a **previsão de atravessar a superquadra vizinha para chegar à superquadra pretendida** (ex.: chegar à SQN 108 passando pela 109): como Brasília preserva o caráter gregário de cada superquadra, entrar na quadra do vizinho exige comportar-se como hóspede — reduzir, dar preferência ao pedestre, usar a faixa quando ela existir, reconhecer a sinalização e respeitar o espaço de convívio. Esse contexto está codificado em `BrasiliaUrbanScaleContext.gs` e alimenta a simulação (ambiente `superquadra`), as atividades de cidadania (`WeeklyDynamics.gs`) e o material pedagógico (`PedagogicalContentManager.gs`).

> Nota de rumo: a infraestrutura de cadastros, perfis, relatórios e permissões existe para **servir** esses eixos. Ao evoluir o sistema, mantenha trânsito + cidadania + regras reconhecíveis + escalas de Brasília como o centro; recursos genéricos de "avaliação educacional" são meios, não o fim.

## Visão geral técnica

O sistema é construído em Google Apps Script e organiza estudantes, professores, turmas, frequência, questionários, rubricas, pontuações, simulações de trânsito e relatórios longitudinais. As avaliações cognitivas, psicomotoras, executivas, pedagógicas e de conformidade legal (CTB/ECA) consolidam diferentes dimensões do desenvolvimento do aluno no contexto do trânsito.

## Componentes principais

`Main.gs` contém o núcleo de entrada do servidor. A navegação da aplicação é apoiada por `ClientRouter.html`, `Sidebar.html`, `Header.html`, `MenuService.gs` e `ResourceLoader.gs`. Existem painéis separados para administração, professores e alunos: `DashboardAdmin.html`, `DashboardProfessor.html` e `DashboardAluno.html`.

Os cadastros centrais passam por `AlunoService.gs`, `StudentProfileManager.gs`, `TeacherProfileManager.gs`, `ClassroomManager.gs`, `SchoolYearManager.gs` e `UserService.gs`. Frequência, grupos, conquistas, feedback e notas possuem gerenciadores próprios. As avaliações são implementadas em `CognitiveEval.gs`, `PsychomotricityEval.gs`, `ExecutiveFuncEval.gs`, `PedagogicalEval.gs` e `LegalFrameworkEval.gs`. As dinâmicas semanais estão distribuídas em `Semana1Logic.gs` até `Semana4Logic.gs` e `WeeklyDynamics.gs`.

`GamificationEngine.gs` concentra pontos, níveis, ranking, distintivos e situações gamificadas. Para cenários de trânsito situados em Brasília, use `getTrafficGameSituations()`, `getTrafficGameSituationsForWeek(weekNumber)` e `awardTrafficSituation(alunoId, situationId, evidence, reason)`. As situações cobrem faixa de entrequadra, quadra vizinha, eixinho/tesourinha, Eixo Monumental, rota Asa Norte/Asa Sul, ponto de ônibus/comércio local e entrada/saída escolar na superquadra.

O painel homologado do aluno também executa uma missão curta de decisão cidadã. O ciclo não termina no “acertou/errou”: `CitizenDecisionService.gs` exige escolha e justificativa, mostra a consequência e só marca a atividade como concluída depois que o estudante registra uma revisão e uma próxima ação concreta. As chamadas `Citizenship.scenario`, `Citizenship.submit` e `Citizenship.reflect` passam pelo gateway autenticado, e a autoria da tentativa é conferida no servidor.

## Dados e relatórios

O armazenamento é orientado a planilhas Google, com suporte de `SpreadsheetUtils.gs`, serviços de importação e exportação e regras de validação. IDs de planilhas, pastas e parâmetros devem ser definidos por `ConfigurationManager.gs`, `ConfigService.gs` e propriedades do script. Não registre credenciais diretamente nos fontes.

Relatórios são tratados por `RelatorioService.gs`, `AdvancedReporting.gs`, `SimulationReportGenerator.gs`, `ReportTemplateManager.gs`, `ReportScheduler.gs`, `PDFGenerator.gs` e diversas telas HTML. Antes de gerar PDFs ou enviar comunicações, confirme as permissões do Drive e do Gmail e valide o volume para não exceder cotas.

## Boas práticas de codificação

- Mantenha controladores e serviços orientados a casos de uso. A interface deve acessar uma fachada pequena e estável, centralizando chamadas `google.script.run`, carregamento, falhas e mensagens ao usuário.
- Padronize retornos como `{ ok, data, error, meta }` e não devolva exceções brutas. Erros de validação, permissão e infraestrutura devem possuir códigos distintos.
- Use `const` por padrão, `let` apenas para valores mutáveis e nomes consistentes em `camelCase`, `PascalCase` e `UPPER_SNAKE_CASE`. Evite criar novas variações de nomes em português e inglês para o mesmo conceito.
- Acrescente JSDoc às funções públicas, regras de avaliação e formatos de relatório. Prefira funções curtas e componíveis, com retornos antecipados e sem efeitos colaterais ocultos.
- Centralize regras pedagógicas nos módulos apropriados. Uma fórmula de pontuação ou critério de avaliação não deve ser repetida no HTML, no relatório e no serviço.
- Leia e grave planilhas em lote; use bloqueios em alterações concorrentes de frequência, notas e pontuação. Scripts agendados devem ser idempotentes para que uma repetição não duplique registros ou mensagens.
- Faça validação e autorização no servidor, incluindo vínculo entre usuário, turma e estudante. Escape conteúdo inserido em HTML e nunca construa marcação com valores não higienizados.
- Registre operação, usuário técnico, duração e resultado sem incluir avaliações sensíveis completas. Defina política de retenção para logs, backups e relatórios exportados.
- Teste cálculos com limites e dados ausentes, permissões por perfil, reexecução de gatilhos, geração de PDF e compatibilidade de versões do esquema.

## Implantação

Crie o projeto em `script.google.com`, envie os arquivos, revise `appsscript.json` e configure o fuso horário. Cadastre propriedades e recursos externos, execute a inicialização necessária e conceda as autorizações solicitadas. Em seguida, publique como aplicativo da Web. Para dados escolares, prefira acesso restrito ao domínio e revise a opção “executar como”.

## Segurança e operação

`AuthService.gs`, `PermissionService.gs`, `UserRoleManager.gs` e `SessionManager.gs` controlam autenticação e autorização. Toda entrada deve atravessar `FormValidationService.gs`, `ValidationUtils.gs` e `DataValidationRules.gs`. Use `AuditLogService.gs`, `EventLogger.gs`, `Logger.gs` e `UserActivityLog.gs` para rastreabilidade. `CacheManager.gs`, `BackupRestoreService.gs`, `SystemHealthMonitor.gs` e `ErrorHandling.gs` apoiam continuidade operacional.

Consulte `Documentacao_Arquitetura.md`, `architecture_plan.md` e os relatórios de maturidade antes de mudanças estruturais. Teste especialmente cadastros, perfis, avaliações, cálculo de pontuação, relatórios, notificações e permissões por papel.


---

## Mapeamento de Schema da Planilha (item 6 — pré-requisito para fixtures analíticos)

> **Status do catálogo AI:** vazio — o `SchemaService` expõe apenas abas de infraestrutura baseline. Nenhuma entidade do domínio de educação para o trânsito está mapeada.

### Abas declaradas no SchemaService

| Aba (sheetName) | Entidade | Tipo | Colunas |
|---|---|---|---|
| `Usuarios` | USERS (login real e baseline) | Autenticação | `ID`, `Username`, `Password`, `Role`, `Nome`, `Email`, `Status`, `LastLoginAt`, `CriadoEm`, `AtualizadoEm` |
| `Settings` | SETTINGS | Configuração/Infra | `Key`, `Value`, `Description`, `Scope`, `UpdatedAt`, `UpdatedBy` |
| `Audit_Logs` | AUDIT_LOGS | Infraestrutura | `ID`, `Timestamp`, `Level`, `Action`, `Entity`, `RecordID`, `UserID`, `Message`, `Details`, `CreatedAt` |

> **Nota:** Este projeto tem a migração de abas legadas mais elaborada da frota — `migrateLegacyUserSheets()` consolida `Users` + `Usuarios` e aposenta `YOUR_SPREADSHEET_ID_HERE` (placeholder que virou aba real). Há também integração com `seedTrafficBadges()` — indicando a existência de entidade `Badges` de trânsito/cidadania.

### Entidades inferidas (não declaradas no SchemaService)

O projeto tem um sistema de gamificação de trânsito (distintivos, desafios, conquistas) que **não está no schema**:

| Entidade inferida | Evidência | O que precisa ser feito |
|---|---|---|
| Badges / Distintivos de trânsito | `seedTrafficBadges()` chamado em `runSchemaServiceSetup` | Declarar entidade `BADGES` com tipo, critério, ícone, pontos |
| Desafios / Atividades | Domínio "brincando no trânsito" implica atividades pedagógicas | Identificar aba real e declarar |
| Progresso / Conquistas do aluno | Gamificação implica FK aluno → badge ganho | Declarar entidade `STUDENT_BADGES` ou `CONQUISTAS` |
| Regras de trânsito / Conteúdo | Base de conhecimento do jogo | Identificar aba e declarar entidade de conteúdo |

### Entidades pendentes de mapeamento analítico

| Entidade | Por que ausente | O que precisa ser feito |
|---|---|---|
| `Badges` | `seedTrafficBadges()` existe mas não está no schema | Declarar entidade no `SchemaService` com colunas reais |
| Progresso / Conquistas | Não declarada | Mapear aba real e declarar |
| Atividades pedagógicas | Não declarada | Identificar estrutura das atividades de trânsito no projeto |

> **Ação necessária para o item 6:** Inspecionar as abas reais da planilha (especialmente as criadas por `seedTrafficBadges()`) e declarar todas as entidades de domínio no `SchemaService` antes de criar fixtures analíticos. A presença de `migrateLegacyUserSheets()` indica que a planilha tem histórico de inconsistências — validar a estrutura atual antes de prosseguir.
