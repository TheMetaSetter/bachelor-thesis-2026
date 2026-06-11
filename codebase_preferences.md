# Codebase Preferences for `bachelor-thesis-2026`

## 1. Reference mechanisms and source code alignment

### For data loaders
1. Data loaders need to follow mechanisms under TSLib codebase. Refer to code under `bsc-thesis-ref-codebases/Time-Series-Library`.
2. Synthetic anomaly injection needs to follow mechanisms under CARLA codebase and RedLamp codebase. Refer to code under `bsc-thesis-ref-codebases/CARLA-main/data/augment.py` and `bsc-thesis-ref-codebases/RedLamp/loaders/loader_aug.py`.

### For discrete prototypes or codebook
- Discrete prototypes or codebook follow mechanisms under DALL-E codebase. Refer to code under `bsc-thesis-ref-codebases/DALL-E-master/dall_e`.

---

## 2. Logging and task organization

- Place logs under `documents/logs/MM-DD-YYYY/<task_type>` with `task_type` being one of the four following: `research`, `plan`, `structure`, or `detail`.
    - Research is following prompt under `prompts/1_research_prompt.md`
    - Plan is following prompt under `prompts/2_plan_prompt.md`
    - Structure is following prompt under `prompts/3_structure_prompt.md`
    - Detail is following prompt under `prompts/4_detail_prompt.md`

- Use Weights & Biases to log artifacts and statistics from experiments, and history of experiments, to make sure each experiment in the history can be reproduced easily without effort.

---

## 3. Planning and codebase workflow order

- `plan` to use which design patterns before `structure`.

- Traverse the codebase using the directory tree inside `documents/design/design_starter.md`. This tree is planned to be fixed most of the time. But if you really need to change the structure of the directory, please remember to update the tree.

---

## 4. Model file organization and self-contained design

- Keep everything related to one particular model placed inside 1 single file for that model.

- Stick to `1 model - 1 file` rule.

- All inference logic and training logic of one model need to be placed in one single file of that model, in a way such that user can read the easiest.

- Core logic of one model, including inference and training logic, should be well-written and can be read from top-to-bottom in ONE SINGLE FILE of that model. The purpose is to make this self-contained.

- All calculations directly related to one model need to be placed inside the single file of that model, or more ideally, to be placed within Python classes of that model.

- All logic related to one model needs to be placed in one single file of that model.

---

## 5. Readability-first principles

This is important so I will repeat 3 times.

- In research codebase like this one, READABILITY IS KING. So you can DO repeat yourself in a thoughtful way to make users read the code with ease. Variables should be named explicitly, with full words, even several words, readability is primordial.

- In research codebase like this one, READABILITY IS KING. So you can DO repeat yourself in a thoughtful way to make users read the code with ease. Variables should be named explicitly, with full words, even several words, readability is primordial.

- In research codebase like this one, READABILITY IS KING. So you can DO repeat yourself in a thoughtful way to make users read the code with ease. Variables should be named explicitly, with full words, even several words, readability is primordial.

- In this codebase, readability is king. So, DO repeat yourself in a thoughtful way.

- Stick to "least amount of codepaths" principle, which means configurations, models, pre-processing, all of that should be obvious to users. Reading should be obvious and configurations should be obvious.

- Do not write overly long `for` loops. When a loop starts handling many responsibilities at once, split the work into small, clearly named helper functions or methods so each step can be read, reused, tested, and modified independently.

- Write explanatory comments to support user reading code. Comments should be updated in parallel with code or implementations.

- Write explanatory comments to support user reading code. Comments should be updated in parallel with code or implementations.

- When writing, reviewing, or refactoring Python code in this repository, strictly follow `python_semantics.md` as a required semantic checklist for slicing, indexing, assignment, mutability, and function-call behavior.

---

## 6. Data versioning and reproducibility

- Each time we do synthetic anomaly injection on one particular dataset will result in a different version of that particular dataset, such as SMD.

- So, please use data version control techniques, use `dvc.yaml`, to control data version, and make sure all history experiments can be reproduced without effort.

---

## 7. Testing requirements

- Add small and minimalistic test cases to check for these things:
    - Input and output tensors of calculations, computations
    - Small test to run one forward pass and one backward pass on one batch of one specific dataset, such as SMD, or let the user choose.
    - TEST THE MECHANISM TO SAVE AND LOAD CHECKPOINTS. THIS IS SUPER IMPORTANT.
    - Test data loaders to make sure the trainers receive the right batch size for each dataset, and the right shapes of tensors.
    - Allow user to test injecting synthetic anomalies by injecting anomalies in one batch of sample in one specific dataset. Then, visualize the injected samples clearly to allow user easily examine the quality of injection.
    - Test the functions or methods within classes that are used to initialize configurations from definition files, such as `.yaml` files. Refer to format under `bsc-thesis-ref-codebases/CARLA-main/configs`. These are just examples; you do not need to do the exact same thing.
    - Design and implement simple test cases using Pytest

---

## 8. Ablation study friendliness

- Design the codebase such that components (e.g., modules, loss terms, preprocessing steps, augmentations) can be easily turned on or turned off without modifying core logic.

- Each component should be controllable via clear and explicit configuration (e.g., `.yaml`), avoiding hidden dependencies or implicit coupling.

- Avoid hard-coded interactions between components; instead, use modular design so that removing or disabling one component does not break the pipeline.

- Ensure that enabling or disabling components results in minimal changes in codepaths, so that ablation experiments are easy to run, compare, and reproduce.
