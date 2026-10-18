using System;
using System.Globalization;
using System.Linq;
using System.Text;

namespace NN
{
    public class Trainer
    {
        public const int DefaultSeed = 42;
        public const int DefaultEpochs = 10000;
        private readonly DataBase data;
        private readonly int seed;
        private readonly int epochs;

        public Trainer(string inputPath, string outputPath = null,
            int seed = DefaultSeed, int epochs = DefaultEpochs)
        {
            if (epochs <= 0)
                throw new ArgumentException("Epochs must be positive.", nameof(epochs));
            data = new DataBase(inputPath, outputPath);
            this.seed = seed;
            this.epochs = epochs;
        }

        internal Network Train()
        {
            data.Read();
            var network = new Network(data.Topology(), seed);
            // Fixed example order and one seeded weight generator make this demo repeatable.
            for (int epoch = 0; epoch < epochs; epoch++)
                foreach (TrainingExample example in data.Examples)
                {
                    network.feedForward(example.Inputs);
                    network.backPropagation(example.Targets);
                }
            return network;
        }

        public string Run()
        {
            Network network = Train();
            var report = new StringBuilder();
            report.AppendLine($"Seed: {seed}; epochs: {epochs}; topology: {string.Join('-', data.Topology())}");
            report.AppendLine("Activation: tanh; fixed bias: +1; learning rate: 0.20; momentum: 0.001");
            double squaredError = 0;
            int count = 0;
            int correct = 0;
            bool binaryTargets = data.Examples.All(example => example.Targets.All(value => value == 0 || value == 1));
            foreach (TrainingExample example in data.Examples)
            {
                network.feedForward(example.Inputs);
                var results = network.getResults();
                if (results.Any(value => !double.IsFinite(value)))
                    throw new InvalidOperationException("Training produced a non-finite prediction.");
                report.AppendLine($"{Format(example.Inputs)} -> {Format(results)} (target {Format(example.Targets)})");
                for (int i = 0; i < results.Count; i++)
                {
                    squaredError += Math.Pow(example.Targets[i] - results[i], 2);
                    count++;
                    if ((results[i] >= 0.5 ? 1.0 : 0.0) == example.Targets[i]) correct++;
                }
            }
            report.AppendLine($"Training-set RMSE: {Math.Sqrt(squaredError / count).ToString("F6", CultureInfo.InvariantCulture)}");
            if (binaryTargets)
                report.AppendLine($"Binary classifications correct (threshold 0.5): {correct}/{count}");
            data.Write(report);
            return report.ToString();
        }

        private static string Format(System.Collections.Generic.IEnumerable<double> values) =>
            string.Join(" ", values.Select(value => value.ToString("F6", CultureInfo.InvariantCulture)));
    }
}
