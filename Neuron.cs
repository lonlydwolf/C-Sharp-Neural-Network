using System;
using System.Collections.Generic;
using System.Linq;


namespace NN
{
    using Layer = List<Neuron>;
    public struct Connections
    {
        public double Weight { get; set; }
        public double deltaWeight { get; set; }
    }

    public class Neuron
    {

        #region Constructor
        public Neuron(int numberOfOutputs, int neuronIndex, Random random)
        {
            for (int i = 0; i < numberOfOutputs; i++)
            {
                outputWights.Add(new Connections());
                Connections temp = outputWights[i];
                temp.Weight = random.NextDouble() - 0.5;
                outputWights[i] = temp;
            }
            Index = neuronIndex;
            bias = 1.0;
        }
        #endregion

        #region Variables
        private int Index;
        private double outputValue;
        private List<Connections> outputWights = new List<Connections>();
        private readonly double bias;
        private double Gradient;
        internal const double TrainingRate = 0.20;
        internal const double Momentum = 0.001;

        #endregion

        #region Private Functions

        /// <summary>
        /// A function to return a non-liner form of the Neuron output.
        /// </summary>
        /// <param name="x">The sum of A_i*W_i 's of the pervious Layer to this Neuron</param>
        /// <returns>tanh of the value</returns>
        private static double activationFunction(double x)
        {
            return Math.Tanh(x);
        }

        private static double activationFunctionDerivative(double x)
        {
            return 1.0 - (x * x);
        }

        private double sumDOW(Layer nextLayer)
        {
            double sum = 0.0;
            sum = nextLayer.Select((n, i) => outputWights[i].Weight * n.Gradient).Sum();
            return sum;
        }
        #endregion

        #region Public Functions

        /// <summary>
        /// Set the output value of the layer that we are in.
        /// (Wich layer is deffind by the Network class)
        /// </summary>
        /// <param name="perviousLayer">The Layer before the layer that we are adjusting</param>
        public void feedForward(Layer perviousLayer)
        {
            double sum = 0.0;
            sum = perviousLayer.Select(N => N.outputValue * N.getOutputWeightOf(this.Index)).Sum();
            sum += bias;
            outputValue = activationFunction(sum);
        }

        /// <summary>
        /// The Error of:
        /// Diffrence between the value that the NN reached
        /// and the value that we aspected.
        /// </summary>
        /// <param name="targetValue">The Value that we aspect</param>
        public void calculateOutputGradients(double targetValue)
        {
            double delta = targetValue - outputValue;
            Gradient = delta * activationFunctionDerivative(outputValue);
        }

        /// <summary>
        /// How much each Neuron effects the output
        /// </summary>
        /// <param name="nextLayer"></param>
        public void calculateHiddenGradients(Layer nextLayer)
        {
            double dow = sumDOW(nextLayer);
            this.Gradient = dow * activationFunctionDerivative(outputValue);
        }

        /// <summary>
        /// Update each Neuron connection to a certain Neuron based on
        /// the impact of error of that connection weight to the output value
        /// </summary>
        /// <param name="perviousLayer">all the pervious layer Neurons that are connected to this Nouron</param>
        public void updateWeights(Layer perviousLayer)
        {
            foreach (Neuron neuron in perviousLayer)
            {
                // Preserve the original convention: store the unscaled update, then apply the learning rate.
                double oldDeltaWeight = neuron.outputWights[Index].deltaWeight;
                double newDeltaWeight = neuron.OutPutvalue * Gradient + Momentum * oldDeltaWeight;
                Connections temp = neuron.outputWights[Index];
                temp.deltaWeight = newDeltaWeight;
                temp.Weight += newDeltaWeight * TrainingRate;
                neuron.outputWights[Index] = temp;
            }
        }
        #endregion

        #region Gets & Sets
        public double OutPutvalue { get => outputValue; set { outputValue = value; } }

        private double getOutputWeightOf(int n) => outputWights[n].Weight;
        #endregion
    }
}
