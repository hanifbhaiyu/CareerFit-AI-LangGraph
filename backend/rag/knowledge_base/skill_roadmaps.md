# Skill Roadmaps for Common Technology Roles

Each roadmap below lists what to learn in order, roughly how long it takes with consistent part-time study, and the portfolio project that proves the skill to an employer. Time estimates assume ten to fifteen hours a week and prior programming familiarity.

## Backend developer

Order matters here because each layer depends on the previous one.

1. **One language to a professional standard** (2-3 months). Python, Java, or PHP are all widely hired locally. Depth in one beats familiarity with three.
2. **Relational databases and SQL** (1 month). Schema design, joins, indexing, transactions. Most backend interview failures happen here.
3. **HTTP and REST API design** (3-4 weeks). Methods, status codes, authentication, versioning, pagination.
4. **A web framework** (1 month). FastAPI or Django for Python, Laravel for PHP, Spring Boot for Java.
5. **Testing** (2-3 weeks). Unit and integration tests. Employers notice a repository with tests and read it as professional maturity.
6. **Deployment basics** (3-4 weeks). Docker, environment configuration, a cloud host, CI that runs on push.

Portfolio project: an API with authentication, a real database schema of at least six related tables, tests, and a deployed public URL. A task manager is acceptable; something with genuine domain complexity such as inventory or booking is better.

## Frontend developer

1. **HTML, CSS and the box model properly** (1 month). Skipping this produces developers who can only work inside a component library.
2. **JavaScript fundamentals** (2 months). Closures, promises, async/await, array methods, the event loop.
3. **React** (1-2 months). Components, hooks, state, effects, and when not to use an effect.
4. **State management and data fetching** (3 weeks).
5. **TypeScript** (3-4 weeks). Increasingly assumed rather than optional in listings.
6. **Accessibility and responsive design** (2 weeks). A differentiator, since few junior candidates can discuss it.

Portfolio project: an application that consumes a real API, handles loading and error states properly, works on mobile, and is deployed. Static portfolio pages do not demonstrate enough.

## Data analyst

1. **SQL to an advanced level** (1-2 months). Window functions, CTEs, query optimisation. This is the core hiring filter.
2. **Spreadsheet fluency** (2 weeks). Still the working tool at most employers.
3. **Python with pandas** (1-2 months). Cleaning, joining, aggregating real messy data.
4. **Visualisation and dashboards** (3-4 weeks). Power BI or Tableau, plus one code-based library.
5. **Statistics fundamentals** (1 month). Distributions, correlation versus causation, significance, sampling error.
6. **Communicating findings** (ongoing). The skill that separates analysts who advance from those who do not.

Portfolio project: an end-to-end analysis of a public dataset with a written conclusion that makes a recommendation, not just charts. The written interpretation is what employers actually evaluate.

## Machine learning engineer

Requires a working software engineering foundation first. Attempting this route without it is the most common cause of stalled transitions.

1. **Python, Git, and testing** (assumed prerequisite).
2. **Mathematics foundations** (2-3 months). Linear algebra, probability, calculus for optimisation. Enough to read a paper, not to prove theorems.
3. **Classical machine learning** (2 months). Regression, trees, ensembles, cross-validation, and why a model that scores well can still be useless.
4. **Deep learning** (2-3 months). One framework, in depth.
5. **Deployment and MLOps** (2 months). Serving a model behind an API, monitoring drift, versioning data and models. This is where demand currently outstrips supply.
6. **Domain specialisation** (ongoing). Language, vision, or tabular forecasting.

Portfolio project: a deployed model with a live inference endpoint, a documented evaluation of its failure cases, and honest reporting of its limitations. Notebooks alone are not enough; employers hire people who can ship.

## DevOps and cloud engineer

1. **Linux and shell** (1-2 months). Not optional at any level.
2. **Networking fundamentals** (1 month). DNS, TLS, load balancing, subnets.
3. **Containers** (1 month). Docker, then orchestration concepts.
4. **One cloud provider in depth** (2-3 months). Compute, storage, networking, identity and access management. A certification here is one of the few that recruiters genuinely filter on.
5. **Infrastructure as code** (1 month). Terraform.
6. **CI/CD and observability** (1-2 months). Pipelines, logging, metrics, alerting.

Portfolio project: a reproducible infrastructure repository that provisions a working environment from scratch, with a pipeline that deploys an application into it.

## Choosing between roadmaps

Pick based on what the local market is hiring for and what you can sustain for six months, in that order. A roadmap you abandon at month three has negative value, because the time is gone and there is no portfolio artefact to show for it.

Learn in the order given. Skipping foundations produces candidates who can complete a tutorial but fail the first interview question that steps outside it.
