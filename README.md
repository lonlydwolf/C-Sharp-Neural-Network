# C# neural network: a completed XOR exercise

A small neural network implemented with individual neurons and connections, without an external ML library or matrix-based implementation. This completed learning exercise is retained for study; there is no active feature roadmap or ongoing support commitment.

The reproducibility closeout adds a bundled XOR dataset, seeded initialization, input validation and regression checks. It also corrects the multi-output error calculation, which previously overwrote each output's squared error instead of summing them. The original fixed bias and weight-update convention remain intact.

## Run

Install a [.NET 10 SDK](https://dotnet.microsoft.com/en-us/download/dotnet/10.0). .NET 10 is an [LTS release supported through November 2028](https://dotnet.microsoft.com/en-us/platform/support/policy).

From this repository:

```sh
dotnet build NN.csproj --configuration Release
dotnet run --project NN.csproj --configuration Release --no-build
dotnet run --project NN.csproj --configuration Release --no-build -- --test
```

The default run prints the four XOR predictions, training-set RMSE and classification count. It reads the bundled fixture relative to the executable, so launching the built DLL from another working directory also works. It writes no output file unless requested.

```sh
dotnet run --project NN.csproj --configuration Release -- --seed 7 --epochs 10000
dotnet run --project NN.csproj --configuration Release -- --output artifacts/xor-report.txt
dotnet run --project NN.csproj --configuration Release -- --input Data/xor.txt --output artifacts/custom-report.txt
```

Custom paths are relative to the invoking working directory. Missing report directories are created; using the same normalized input/output path is rejected. Invalid arguments or data return a nonzero exit code with an error message. `--help` lists the options.

## Algorithm and experiment boundaries

- Topology: **2–4–1**; activation: **tanh**.
- Each non-input neuron adds a **fixed +1 bias**. Biases are not trained.
- Learning rate: **0.20**; momentum: **0.001**. The code stores an unscaled connection update, then multiplies it by the learning rate when changing the weight.
- Default seed: **42**, from one shared `Random` instance. Initial weights are in `[-0.5, 0.5)`.
- Default training: **10,000 epochs** over the four examples in fixed order, or **40,000 online updates**. An epoch means one traversal of the dataset.
- Targets are 0/1. A threshold of 0.5 determines the displayed binary classifications; raw outputs can be negative because the activation is tanh.

This is a training-set convergence demonstration, not a generalization benchmark. The smoothed per-example error is not a rolling-window average. Seeded runs are repeatable in the tested runtime; exact floating-point results across SDKs/platforms and numerical parity with the C++ or other versions are not promised.

## Dataset format

The final integer on the first line is the example count; preceding integers are layer widths. Each example has one input line and one target line:

```text
Topology: 2 4 1 1
In: 0.0 1.0
Target: 1.0
```

Historical repeated labels such as `In: 0.0 In: 1.0` are also accepted. At least two positive layer widths and a positive example count are required. Input/target widths must match the first/last layer, and the count must match all pairs. Numbers use `.` as the decimal separator regardless of locale; non-finite values are rejected. The four-example fixture is in [Data/xor.txt](Data/xor.txt).

## Checks and structure

`--test` runs dependency-free regression checks and fails with a nonzero exit code. It covers all four XOR classifications and RMSE below 0.1 for seeds 42, 7 and 123; repeatability; numerical gradients and momentum for all 12 connections of a two-output test network over two updates; multi-output RMSE; malformed data; locale-independent parsing; missing files; and report-directory creation. GitHub Actions runs the build, checks and demo on Linux.

`Network` traverses the layers, `Neuron` calculates outputs and updates, `DataBase` handles the text fixture, and `Trainer` performs repeated passes and reports final predictions. The historical unused NUnit package has been removed; no external packages are required.

The archived [XOR-First-Principles-Network parent](https://github.com/lonlydwolf/XOR-First-Principles-Network) pins a historical revision of this repository. These closeout fixes are in the direct C# repository; cloning the parent's old submodule revision will not include them.
