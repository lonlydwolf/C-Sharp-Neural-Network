using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;

namespace NN
{
    // A small dependency-free regression runner; --test returns nonzero on failure.
    internal static class RegressionTests
    {
        public static int Run()
        {
            var tests = new (string Name, Action Test)[]
            {
                ("seeded XOR convergence and repeatability", Xor),
                ("all 12 gradients, momentum and multi-output RMSE", Gradients),
                ("dataset validation, invariant parsing and output paths", DataFiles),
                ("network dimensions and finite inputs", InvalidNetworkInputs)
            };
            int failed = 0;
            foreach (var test in tests)
            {
                try { test.Test(); Console.WriteLine($"PASS: {test.Name}"); }
                catch (Exception error) { failed++; Console.Error.WriteLine($"FAIL: {test.Name}: {error.Message}"); }
            }
            Console.WriteLine($"{tests.Length - failed}/{tests.Length} regression groups passed.");
            return failed == 0 ? 0 : 1;
        }

        private static void Xor()
        {
            string path = Path.Combine(AppContext.BaseDirectory, "Data", "xor.txt");
            string first = new Trainer(path).Run();
            Equal(first, new Trainer(path).Run(), "same seed must reproduce the report");
            foreach (int seed in new[] { 42, 7, 123 })
            {
                Network network = new Trainer(path, seed: seed).Train();
                var inputs = new[] { new List<double> { 0, 0 }, new List<double> { 0, 1 },
                    new List<double> { 1, 0 }, new List<double> { 1, 1 } };
                double[] expected = { 0, 1, 1, 0 };
                double squaredError = 0;
                for (int i = 0; i < inputs.Length; i++)
                {
                    network.feedForward(inputs[i]);
                    double value = network.getResults()[0];
                    Check(double.IsFinite(value), "prediction must be finite");
                    Equal(expected[i], value >= 0.5 ? 1.0 : 0.0, $"XOR classification, seed {seed}");
                    squaredError += Math.Pow(expected[i] - value, 2);
                }
                Check(Math.Sqrt(squaredError / 4) < 0.1, $"XOR RMSE must be below 0.1, seed {seed}");
            }
        }

        private static void Gradients()
        {
            var network = new Network(new List<int> { 2, 3, 2 }, 7);
            var input = new List<double> { 0.3, -0.7 };
            var target = new List<double> { 0.2, -0.4 };
            var edges = Connections(network).ToArray();
            Equal(12, edges.Length, "all connections must be checked");
            double[] previousUpdates = new double[edges.Length];
            double average = 0;
            for (int step = 0; step < 2; step++)
            {
                double[] before = edges.Select(edge => edge.List[edge.Index].Weight).ToArray();
                double[] gradients = new double[edges.Length];
                const double epsilon = 1e-6;
                for (int i = 0; i < edges.Length; i++)
                {
                    SetWeight(edges[i], before[i] + epsilon);
                    double plus = Loss(network, input, target);
                    SetWeight(edges[i], before[i] - epsilon);
                    double minus = Loss(network, input, target);
                    SetWeight(edges[i], before[i]);
                    gradients[i] = (plus - minus) / (2 * epsilon);
                    Check(double.IsFinite(gradients[i]) && Math.Abs(gradients[i]) > 1e-9,
                        $"edge {i} needs a finite, nonzero numerical gradient");
                }
                double loss = Loss(network, input, target);
                double rmse = Math.Sqrt(2 * loss / target.Count);
                network.backPropagation(target);
                Near(rmse, network.LastError, 1e-12, "multi-output RMSE before update");
                average = (average * 100 + rmse) / 101;
                Near(average, network.AverageError, 1e-12, "smoothed RMSE");
                for (int i = 0; i < edges.Length; i++)
                {
                    double update = edges[i].List[edges[i].Index].Weight - before[i];
                    double expected = -0.20 * gradients[i] + 0.001 * previousUpdates[i];
                    Near(expected, update, 1e-8, $"gradient/momentum update for edge {i}, step {step}");
                    previousUpdates[i] = update;
                }
            }
        }

        private static double Loss(Network network, List<double> input, List<double> target)
        {
            network.feedForward(input);
            return network.getResults().Select((value, i) => 0.5 * Math.Pow(target[i] - value, 2)).Sum();
        }

        // Inspect weights only in the regression runner; production does not expose mutation hooks.
        private static IEnumerable<(List<Connections> List, int Index)> Connections(Network network)
        {
            var layers = (List<List<Neuron>>)typeof(Network)
                .GetField("Layers", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(network);
            foreach (var layer in layers)
                foreach (var neuron in layer)
                {
                    var connections = (List<Connections>)typeof(Neuron)
                        .GetField("outputWights", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(neuron);
                    for (int i = 0; i < connections.Count; i++)
                        yield return (connections, i);
                }
        }

        private static void SetWeight((List<Connections> List, int Index) edge, double value)
        {
            Connections connection = edge.List[edge.Index];
            connection.Weight = value;
            edge.List[edge.Index] = connection;
        }

        private static void DataFiles()
        {
            string directory = Path.Combine(Path.GetTempPath(), "xor-regression-" + Guid.NewGuid());
            Directory.CreateDirectory(directory);
            CultureInfo original = CultureInfo.CurrentCulture;
            try
            {
                CultureInfo.CurrentCulture = CultureInfo.GetCultureInfo("fr-FR");
                string input = Path.Combine(directory, "input.txt");
                string output = Path.Combine(directory, "nested", "report.txt");
                File.WriteAllText(input, "Topology: 2 3 2 1\nIn: 0.25 -0.5\nTarget: 1.0 0.0\n");
                Throws<ArgumentException>(() => new DataBase(input, Path.Combine(directory, ".", "input.txt")));
                var data = new DataBase(input, output);
                data.Read();
                Equal(1, data.NumberOfTestCases, "example count");
                Equal(0.25, data.Examples[0].Inputs[0], "decimal parsing under fr-FR");
                Equal(2, data.Examples[0].Targets.Count, "multi-output fixture");
                File.WriteAllText(input, "Topology: 2 3 2 1\nIn: 0.25 In: -0.5\nTarget: 1.0 Target: 0.0\n");
                data.Read();
                Equal(-0.5, data.Examples[0].Inputs[1], "historical repeated input labels");
                Equal(0.0, data.Examples[0].Targets[1], "historical repeated target labels");
                data.Write(new StringBuilder("report"));
                Equal("report", File.ReadAllText(output), "missing output parent created");
                Throws<FileNotFoundException>(() => new DataBase(Path.Combine(directory, "missing")).Read());
                string[] invalid = {
                    "", "Topology: 2 0 1 1\nIn: 0 0\nTarget: 0\n",
                    "Topology: 2 1 2\nIn: 0 0\nTarget: 0\n",
                    "Topology: 2 1 1\nIn: 0\nTarget: 0\n",
                    "Topology: 2 2 1\nIn: 0 1\nTarget: 0\n",
                    "Topology: 2 1 1\nIn: NaN 0\nTarget: 0\n",
                    "Topology: 2 1 1\nIn: 0 0\nTarget: Infinity\n",
                    "Topology: 2 1 1\nIn: 0 0\nTarget: 0,5\n",
                    "Topology: 2 1 1\nOther: 0 0\nTarget: 0\n"
                };
                foreach (string contents in invalid)
                {
                    File.WriteAllText(input, contents);
                    Throws<InvalidDataException>(() => new DataBase(input).Read());
                }
            }
            finally { CultureInfo.CurrentCulture = original; Directory.Delete(directory, true); }
        }

        private static void InvalidNetworkInputs()
        {
            Throws<ArgumentException>(() => new Network(new List<int> { 2 }));
            Throws<ArgumentException>(() => new Network(new List<int> { 2, 0, 1 }));
            var network = new Network(new List<int> { 2, 2, 1 });
            Throws<InvalidOperationException>(() => network.feedForward(new List<double> { 1 }));
            Throws<InvalidOperationException>(() => network.feedForward(new List<double> { double.NaN, 1 }));
            Throws<ArgumentException>(() => network.backPropagation(new List<double> { 0, 1 }));
            Throws<ArgumentException>(() => network.backPropagation(new List<double> { double.PositiveInfinity }));
            Throws<ArgumentException>(() => new Trainer("unused", epochs: 0));
        }

        private static void Throws<T>(Action action) where T : Exception
        {
            try { action(); }
            catch (T) { return; }
            throw new Exception($"Expected {typeof(T).Name}.");
        }
        private static void Check(bool condition, string message)
        {
            if (!condition) throw new Exception(message);
        }
        private static void Equal<T>(T expected, T actual, string message) =>
            Check(EqualityComparer<T>.Default.Equals(expected, actual), $"{message}: expected {expected}, got {actual}");
        private static void Near(double expected, double actual, double tolerance, string message) =>
            Check(double.IsFinite(actual) && Math.Abs(expected - actual) <= tolerance,
                $"{message}: expected {expected}, got {actual}");
    }
}
