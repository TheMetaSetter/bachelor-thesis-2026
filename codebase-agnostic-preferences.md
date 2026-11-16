# Codebase-Agnostic Preferences

## Scope and evidence

These preferences describe
reusable working preferences, not
universal software rules.

Confidence: **High** for preferences stated directly in the source;
**Medium** for applying model-specific rules to other stateful components.
Research and machine-learning rules below apply only when relevant.

## 1. Put readability first

- Prefer the smallest workable design that preserves required behavior.
- Use explicit, descriptive names with full words. A longer name is acceptable
  when it makes the meaning clearer.
- Allow thoughtful repetition when it makes code easier to read. Avoid
  abstractions that make readers jump between files to understand a simple step.
- Keep the number of execution paths small. Make configuration, preprocessing,
  and component behavior obvious.
- Give each term one meaning and each runtime concept one canonical name.
- Keep every function and method at **50 lines or fewer**.
- Keep every code file at **500 lines or fewer**. Move non-core helpers into
  focused files when necessary.
- Split loops that handle several responsibilities into short, clearly named
  steps. Short code must still have an understandable runtime flow.

## 2. Make code teachable

Use **ASD-STE100 (Simplified Technical English)** for English technical writing
in all projects. Apply it to documentation, specifications, comments, code
reviews, explanations, and command-line help.

- Follow the standard's writing rules and controlled dictionary.
- Use short, direct sentences and consistent terms.
- Use the standard's provisions for technical names and technical verbs when
  software terms are necessary. Keep exact code identifiers and API names.
- Use the [official ASD-STE100 source](https://www.asd-ste100.org/) when checking
  the rules. Simple wording alone does not establish full compliance.

Write code, reviews, comments, and explanations so a high-school student can
follow the intent, data flow, and runtime contract without guessing.

- Explain the surrounding computation from first principles when changing code.
- Use short comments to explain what each important step does and why it exists.
- Update comments together with implementation changes.
- Add small flow diagrams when they clarify where a snippet fits in the full
  processing pipeline.
- If the behavior cannot be explained clearly, simplify the implementation or
  its explanation before calling the work complete.

## 3. Keep ownership and public interfaces clear

For models, keep exactly one public entrypoint per model. That entrypoint owns
construction, public training and inference methods, configuration, and the
checkpoint contract. Apply the same ownership principle to comparable stateful
components when appropriate.

- Let readers start at the public entrypoint and follow the full runtime flow
  through explicitly named helpers.
- Extract helpers for small reusable operations or immutable configuration.
- Avoid mixins that spread a component's lifecycle across files.
- Do not hide phase-specific behavior or introduce competing public
  implementations inside helper files.

## 4. Research, plan, then structure

- At every implementation attempt, first look for code that already implements
  the required behavior. Check the current codebase, its dependencies, and
  relevant external implementations before writing new code.
- Give priority to reusing suitable code. Check its behavior, interface,
  license, and dependencies. Write new code when existing code does not meet
  the requirements, and explain that gap briefly.
- Check relevant reference implementations before implementing an established
  mechanism. Preserve the intended behavior rather than merely copying names.
- Choose and explain useful design patterns during planning, before defining
  the directory structure. Do not add patterns without a concrete need.
- Use the project's documented directory tree to navigate the codebase.
- Keep the structure stable where practical. Update its documentation whenever
  the actual directory structure changes.
- Organize work records by date and task type. Use `yyyy-mm-dd` for new
  date-based directories. Retain the categories `research`, `plan`, `structure`,
  and `detail` where this workflow applies.

## 5. Preserve meaning across specification versions

Before writing, reviewing, or implementing a new specification, compare all
relevant earlier versions. The newest document alone is insufficient context.

- Record terminology changes with the old name, new name, semantic status,
  runtime owner, and supporting evidence.
- Use these statuses: `unchanged`, `renamed`, `split`, `merged`, `new`, and
  `deprecated`.
- Check schemas, methods, stored values, lifecycle, ownership, callers,
  decisions affected, and serialization before treating two names as equivalent.
- For a rename, state whether meaning stays the same and identify affected
  source code, configuration, checkpoints, metrics, and artifacts.
- Ask the human developer when semantic identity remains unclear and affects
  planning or implementation. Similar names are insufficient evidence.
- Include a terminology-change section in every new specification version.
  Explicitly state when no names changed.

## 6. Keep configuration explicit and components independent

- Use clear configuration keys to control optional components.
- Allow components to be enabled or disabled without editing core logic.
- Avoid hidden dependencies and hard-coded interactions between components.
- Keep configuration variants on as few execution paths as practical.
- Make disabled-component behavior explicit so comparisons remain meaningful.

For research, this supports ablation studies: experiments that remove or disable
one component to measure its contribution.

### Reuse pipelines through script arguments

Prefer simple Python scripts or shell scripts that users can run again with
different argument values. This is especially important for data preprocessing.

- Accept input paths, output paths, and relevant processing settings as
  arguments. Do not require source edits to process different input files.
- Keep the processing logic reusable. Let scripts call that logic instead of
  making a separate implementation for each file or dataset.
- Run the same preprocessing pipeline on new files by changing argument values.
  Avoid copied scripts and hard-coded paths for individual runs.
- Document the arguments, their defaults, and short example commands. State
  where the script writes its results.
- Keep script arguments explicit and the number of execution paths small.

## 7. Keep outputs small, organized, and traceable

- Define one canonical result hierarchy with clear meanings for its levels.
- Aim for approximately **3 levels below an experiment root**. Allow up to
  **6 levels** when needed; treat 6 as a hard maximum, not the default target.
- Support historical layouts through discovery or adapters without redefining
  the canonical layout.
- Save task-relevant summaries, essential checkpoints, evaluation rules,
  configuration, provenance, and explicitly selected diagnostics.
- Avoid saving large raw tensors, per-pass outputs, traces, intermediate logs,
  tool internals, or duplicate artifacts by default. Require a documented
  analysis or audit reason to retain them.
- Compute intermediate values on demand when practical.

For staged model training, retain the checkpoint that initializes each stage
and the best checkpoint selected by its monitoring rule. Record their paths,
roles, and checksums in the stage manifest. These retention requirements take
precedence over general storage minimization.

## 8. Make research reproducible

- Version transformed or generated datasets. Treat each synthetic-data
  generation run as a distinct dataset version.
- Track experiment history, configuration, inputs, outputs, and provenance so
  earlier results can be reproduced with little manual reconstruction.
- Use tools suited to the project. The source uses DVC for data and pipeline
  versioning and Weights & Biases for experiment tracking; these tool choices
  are not requirements for every codebase.
- Run one concrete combination through the complete experiment flow before
  launching the full experiment set. Broader preliminary coverage may follow
  when time and project instructions allow it.

## 9. Test important contracts with small examples

Prefer small, focused tests that verify observable behavior and important
boundaries. Use a test framework appropriate to the language; the source
prefers Pytest for Python.

For data and machine-learning projects, cover:

- Input and output shapes for important computations.
- One forward pass and one backward pass on a representative batch.
- Checkpoint saving and loading. Treat this as a critical requirement.
- Data-loader batch sizes and tensor shapes.
- Configuration initialization from definition files.
- A small sample of synthetic-data transformations, with clear visual output
  so a person can inspect their quality.

For Python, check slicing, indexing, assignment, mutability, and function-call
behavior against the project's semantic checklist when one exists.

## 10. Make commands inspectable and remote changes precise

- Write short shell commands, preferably one action per line or terminal step.
- Make each command easy to inspect, rerun, and debug independently.
- Bundle commands into a scripted batch only when the user requests it.
- On shared remote machines, inspect the current state before writing or
  cleaning up.
- Scope changes to exact process IDs, run directories, or artifact paths.
- Treat broad recursive deletion as a last resort. Confirm the target is safe
  to remove first.
- Before leaving a shared host, check that unrelated jobs, logs, and temporary
  files were unaffected.

## Interpretation limits

This extraction does not carry over particular datasets, algorithms, reference
repositories, directory trees, terminology documents, or model architectures.
The reusable preference is to define and follow clear project contracts.

The 50-line and 500-line limits are this person's explicit preferences, not
claims about universal software quality. The source's one-combination-only
restriction belongs to its original project; the reusable rule is to complete
a representative end-to-end run before scaling up.

The readability and explicitness preferences also align with
[Python's PEP 20](https://peps.python.org/pep-0020/). That reference provides
context, not evidence of this person's preferences; the supplied text is the
evidence for this extraction.
