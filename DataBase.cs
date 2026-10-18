using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

namespace NN
{
    public record TrainingExample(List<double> Inputs, List<double> Targets);

    // The historical text format ends the topology line with the number of examples.
    public class DataBase
    {
        private readonly string inputFilePath;
        private readonly string outputFilePath;
        private List<int> topology = new List<int>();
        private readonly List<TrainingExample> examples = new List<TrainingExample>();

        public DataBase(string inputPath, string outputPath = null)
        {
            inputFilePath = Path.GetFullPath(inputPath);
            outputFilePath = outputPath == null ? null : Path.GetFullPath(outputPath);
            StringComparison comparison = OperatingSystem.IsWindows() || OperatingSystem.IsMacOS()
                ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
            if (string.Equals(inputFilePath, outputFilePath, comparison))
                throw new ArgumentException("Output must not overwrite the input dataset.", nameof(outputPath));
        }

        public IReadOnlyList<TrainingExample> Examples => examples;
        public int NumberOfTestCases => examples.Count;
        public List<int> Topology() => new List<int>(topology);

        public void Read()
        {
            string[] lines = File.ReadAllLines(inputFilePath)
                .Where(line => !string.IsNullOrWhiteSpace(line)).ToArray();
            if (lines.Length == 0)
                throw new InvalidDataException("Input file is empty.");

            string[] header = Values(lines[0], "Topology:");
            if (header.Length < 3 || header.Any(value => !int.TryParse(value,
                NumberStyles.None, CultureInfo.InvariantCulture, out int n) || n <= 0))
                throw new InvalidDataException("Topology requires positive layer sizes and an example count.");

            int count = int.Parse(header[header.Length - 1], CultureInfo.InvariantCulture);
            List<int> parsedTopology = header.Take(header.Length - 1)
                .Select(value => int.Parse(value, CultureInfo.InvariantCulture)).ToList();
            if (lines.LongLength != 1L + 2L * count)
                throw new InvalidDataException("Example count does not match the input/target pairs.");

            var parsedExamples = new List<TrainingExample>();
            for (int i = 0; i < count; i++)
            {
                var inputs = ParseNumbers(lines[1 + 2 * i], "In:", parsedTopology[0]);
                var targets = ParseNumbers(lines[2 + 2 * i], "Target:", parsedTopology.Last());
                parsedExamples.Add(new TrainingExample(inputs, targets));
            }
            topology = parsedTopology;
            examples.Clear();
            examples.AddRange(parsedExamples);
        }

        private static string[] Values(string line, string prefix)
        {
            line = line.Trim();
            if (!line.StartsWith(prefix, StringComparison.Ordinal))
                throw new InvalidDataException($"Expected '{prefix}' at the start of a data line.");
            return line.Substring(prefix.Length).Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
        }

        private static List<double> ParseNumbers(string line, string prefix, int width)
        {
            // Older generators repeated the label before each value on a line.
            string[] values = Values(line, prefix).Where(value => value != prefix).ToArray();
            if (values.Length != width)
                throw new InvalidDataException($"{prefix} requires {width} values.");
            var numbers = new List<double>();
            foreach (string value in values)
            {
                if (!double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out double number)
                    || !double.IsFinite(number))
                    throw new InvalidDataException($"{prefix} values must be finite numbers using '.' as the decimal separator.");
                numbers.Add(number);
            }
            return numbers;
        }

        public void Write(StringBuilder report)
        {
            if (outputFilePath == null)
                return;
            string fullPath = Path.GetFullPath(outputFilePath);
            Directory.CreateDirectory(Path.GetDirectoryName(fullPath));
            File.WriteAllText(fullPath, report.ToString());
        }
    }
}
