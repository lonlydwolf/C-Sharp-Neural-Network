using System;
using System.Globalization;
using System.IO;

namespace NN
{
    class Program
    {
        static int Main(string[] args)
        {
            try
            {
                if (args.Length == 1 && args[0] == "--test")
                    return RegressionTests.Run();
                if (args.Length == 1 && args[0] == "--help")
                {
                    Console.WriteLine("Usage: NN [--seed INTEGER] [--epochs POSITIVE_INTEGER] [--input FILE] [--output FILE]\n       NN --test");
                    return 0;
                }
                string input = Path.Combine(AppContext.BaseDirectory, "Data", "xor.txt");
                string output = null;
                int seed = Trainer.DefaultSeed;
                int epochs = Trainer.DefaultEpochs;
                for (int i = 0; i < args.Length; i += 2)
                {
                    if (i + 1 >= args.Length)
                        throw new ArgumentException($"Missing value for {args[i]}.");
                    switch (args[i])
                    {
                        case "--input": input = args[i + 1]; break;
                        case "--output": output = args[i + 1]; break;
                        case "--seed": seed = int.Parse(args[i + 1], CultureInfo.InvariantCulture); break;
                        case "--epochs": epochs = int.Parse(args[i + 1], CultureInfo.InvariantCulture); break;
                        default: throw new ArgumentException($"Unknown option: {args[i]}. Use --help.");
                    }
                }
                Console.Write(new Trainer(input, output, seed, epochs).Run());
                return 0;
            }
            catch (Exception error) when (error is ArgumentException || error is IOException
                || error is UnauthorizedAccessException || error is FormatException
                || error is OverflowException || error is InvalidOperationException)
            {
                Console.Error.WriteLine($"Error: {error.Message}");
                return 1;
            }
        }
    }
}
