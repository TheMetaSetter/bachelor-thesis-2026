# Sequence diagram for `/report`

Open an experiment with `/experiment NAME`. Enter `/report` to make tables from the saved results of normal runs. For example, in the `smd_comparison` session, `/report` runs the same command as `tsad report experiment --name smd_comparison`.

```mermaid
sequenceDiagram
    actor User
    participant App as TsadApp
    participant Screen as CommandScreen
    participant CLI as tsad CLI process
    participant Report as report_experiment
    participant Files as Project files

    User->>App: Enter /report
    alt There is no active experiment
        App-->>User: Select an experiment with /experiment NAME.
    else An experiment is active
        App->>Screen: Open command screen for report experiment --name NAME
        Screen->>CLI: Start CLI with the project root
        CLI->>Report: report_experiment(experiment YAML path, project_root)
        Report->>Files: Read experiment YAML and metrics.yaml
        Report->>Files: Find saved runs for this experiment in normal mode
        loop Each reporting profile
            Report->>Files: Read profile and check metric precision
            loop Each dataset with runs in the profile phase
                Report->>Files: Read saved results for the profile splits
                Report->>Report: Check result identity and score file hash if present
                Report->>Report: Select one valid result per seed for each table cell
                Report->>Report: Calculate mean, sample standard deviation, and seed count
                Report->>Files: Write detail CSV, mean CSV, and report metadata JSON
            end
        end
        alt Report command completes
            Report-->>CLI: Return CSV paths
            CLI-->>Screen: Print CSV paths and return exit status 0
        else Report command fails with a handled error
            CLI-->>Screen: Print the error and return exit status 1
        end
        Screen-->>User: Show command output and exit status
        User->>Screen: Select Back or press Esc
        Screen-->>App: Return exit status
        App-->>User: Add /report and its status to the experiment history
    end
```

## Table rules

1. The reporting profile specifies the phase, splits, and metrics. The experiment definition specifies the expected seeds.
2. For each cell, the report uses the most recent successful attempt for each expected seed that has a finite value for that metric. The command stops if the selected attempts have incompatible configurations.
3. The detail CSV shows the mean, sample standard deviation, and the number of valid seeds out of the expected seeds. The mean CSV shows the mean only. The report writes `N/A` for a missing mean or standard deviation.
4. The output directory is `outputs/experiments/NAME/reports/DATASET/`. Each profile writes `PROFILE_detail.csv`, `PROFILE_mean.csv`, and `PROFILE.json`. The command prints the two CSV paths.
5. If no dataset has runs for any selected profile phase, the command returns exit status `0` with no CSV paths. If a later profile or dataset fails, files written earlier can remain.

[Slash command design](tsad-slash-tui-design.md) · [Textual interface](../../../../src/tui.py) · [CLI](../../../../src/cli.py) · [Report code](../../../../src/reporting.py)
