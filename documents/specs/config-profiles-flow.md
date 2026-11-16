# Config profiles and run cell

Status: proposed design for the next version of `bachelor-thesis-2026`.

A run cell selects profiles and provides settings for one run. The resolver combines them into a resolved run configuration.

```mermaid
flowchart LR
    D["Data profile<br/>How to prepare data"]
    M["Method profile<br/>How the method works"]
    E["Evaluation profile<br/>How to score and evaluate"]
    C["Run cell<br/>Selects profiles and run-specific values"]
    D --> C
    M --> C
    E --> C
    C --> R["Resolved run configuration"]
```

The profiles hold settings. The four protocols contain validation rules for inputs and outputs in their respective contexts; they are separate from the profiles.
