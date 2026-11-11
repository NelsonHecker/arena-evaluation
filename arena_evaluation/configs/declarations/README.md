# Declarative configuration

Benchmark-facing YAML lives under `configs/`:

- `configs/benchmark/` — suites, contests, report manifests (plus the run-output fixtures if present).
- `configs/declarations/plot_declarations/` — reusable plot recipes (`PlotSpec` YAML) and the shipped report recipes.
- `configs/declarations/query_declarations/` — reusable SQL query declarations (`QueryDeclaration` YAML) used by `evaluation_query` and the catalog.

Resolution order for both declaration kinds is: explicit caller directories (e.g. a run-local `plot_declarations/` or `query_declarations/`), the repository `configs/declarations/<kind>/`, the working-directory copy, then the installed package share directory (`share/arena_evaluation/configs/declarations/<kind>/`).
