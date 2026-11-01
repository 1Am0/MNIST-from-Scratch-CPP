#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>
#include <vector>
#include <stdexcept>
#include <random>
#include <cmath>
#include <array>
#include <algorithm>

const int INPUTS = 784;
const int HIDDEN = 64;
const int OUTPUTS = 10;
const int BATCH = 32;

std::mt19937 rng(42);

struct Neuron
{
	std::vector<double> w;
	double b = 0;
};

struct Model
{
	std::vector<Neuron> hidden;
	std::vector<Neuron> output;
};

struct ForwardResult
{
	std::array<double, HIDDEN> h;
	std::array<double, OUTPUTS> p;
};

double sigmoid(double z)
{
	return 1 / (1 + std::exp(-z));
}

uint32_t readBigEndian32(std::ifstream& file)
{
	unsigned char buf[4];
	file.read(reinterpret_cast<char*>(buf), 4);

	return static_cast<uint32_t>(buf[0]) << 24
		| static_cast<uint32_t>(buf[1]) << 16 
		| static_cast<uint32_t>(buf[2]) << 8 
		| static_cast<uint32_t>(buf[3]) << 0;
}

std::vector<double> loadImages(const std::string& path, uint32_t& count, uint32_t& rows, uint32_t& cols)
{
	std::ifstream file(path, std::ios::binary);

	if (!file)
	{
		throw std::runtime_error("Could not open " + path);
	}

	uint32_t magic = readBigEndian32(file);
	count = readBigEndian32(file);
	rows = readBigEndian32(file);
	cols = readBigEndian32(file);

	if (magic != 2051)
	{
		throw std::runtime_error("Bad magic number in images file");
	}

	std::vector<unsigned char> raw;
	raw.resize(count * rows * cols);

	file.read(reinterpret_cast<char*>(raw.data()), raw.size());

	if (!file)
	{
		throw std::runtime_error("Pixel data too short in " + path);
	}

	std::vector<double> fOut;
	fOut.resize(count * rows * cols);

	for (size_t i = 0; i < fOut.size(); i++)
	{
		fOut[i] = raw[i] / 255.0;
	}

	return fOut;
}

std::vector<int> loadLabels(const std::string& path, uint32_t& count)
{
	std::ifstream file(path, std::ios::binary);

	if (!file)
	{
		throw std::runtime_error("Could not open " + path);
	}

	uint32_t magic = readBigEndian32(file);
	count = readBigEndian32(file);

	if (magic != 2049)
	{
		throw std::runtime_error("Bad magic number in labels file");
	}

	std::vector<unsigned char> raw;
	raw.resize(count);

	file.read(reinterpret_cast<char*>(raw.data()), raw.size());

	if (!file)
	{
		throw std::runtime_error("Label data too short in " + path);
	}

	std::vector<int> fOut;
	fOut.resize(count);

	for (size_t i = 0; i < fOut.size(); i++)
	{
		fOut[i] = raw[i];
	}

	return fOut;
}

void printDigit(const std::vector<double>& images, const std::vector<int>& labels, int i)
{
	std::cout << "Image label: " << labels[i] << "\n";

	for (int r = 0; r < 28; r++)
	{
		for (int c = 0; c < 28; c++)
		{
			double pixel = images[i * 784 + r * 28 + c];

			std::cout << (pixel >= 0.5 ? "# " : ". ");
		}
		std::cout << "\n";
	}
}

void initNeuron(Neuron& n, int numInputs)
{
	n.w.resize(numInputs);

	double limit = 1 / std::sqrt(numInputs);

	std::uniform_real_distribution<double> randWeight(-limit, limit);

	for (double& weight : n.w)
	{
		weight = randWeight(rng);
	}
}

void initModel(Model& m)
{
	m.hidden.resize(HIDDEN);

	for (Neuron& n : m.hidden)
	{
		initNeuron(n, INPUTS);
	}

	m.output.resize(OUTPUTS);	

	for (Neuron& n : m.output)
	{
		initNeuron(n, HIDDEN);
	}
}

void softmax(std::array<double, OUTPUTS>& v)
{
	double max = v[0];
	for (int i = 1; i < OUTPUTS; i++)
	{
		if (v[i] > max)
		{
			max = v[i];
		}
	}

	double s = 0;

	for (double& val : v)
	{
		val = std::exp(val - max);
		s += val;
	}

	for (double& val : v)
	{
		val /= s;
	}
}

void forward(const Model& m, const double* x, ForwardResult& out)
{
	for (int j = 0; j < HIDDEN; j++)
	{
		const Neuron& n = m.hidden[j];
		double val = n.b;
		for (int i = 0; i < INPUTS; i++)
		{
			val += n.w[i] * x[i];
		}

		double z = sigmoid(val);

		out.h[j] = z;
	}

	for (int j = 0; j < OUTPUTS; j++)
	{
		const Neuron& n = m.output[j];
		double val = n.b;
		for (int i = 0; i < HIDDEN; i++)
		{
			val += n.w[i] * out.h[i];
		}

		out.p[j] = val;
	}

	softmax(out.p);
}

double imageLoss(std::array<double, OUTPUTS> p, int label)
{
	return -std::log(p[label] + 1e-12);
}

double totalLoss(const std::vector<double>& images, const std::vector<int>& labels, const Model& m, int start, int count)
{
	double tLoss = 0;
	
	for (int i = start; i < start + count; i++)
	{
		ForwardResult r;
		
		forward(m, &images[i * INPUTS], r);

		tLoss += imageLoss(r.p, labels[i]);
	}

	return tLoss / count;
}

void zeroNeuron(Neuron& n)
{
	for (double& weight : n.w)
	{
		weight = 0;
	}
	n.b = 0;
}

Model zeroModel(Model m)
{
	for (Neuron& n : m.hidden)
	{
		zeroNeuron(n);
	}
	for (Neuron& n : m.output)
	{
		zeroNeuron(n);
	}

	return m;
}

Model computeGradient(const Model& m, const std::vector<double>& images, const std::vector<int>& labels, int start, int count, const std::vector<int>& order)
{
	Model g = zeroModel(m);
	
	ForwardResult r;

	for (int idx = start; idx < start + count; idx++)
	{
		int i = order[idx];

		forward(m, &images[i * INPUTS], r);

		std::array<double, OUTPUTS> eOut;

		for (int k = 0; k < OUTPUTS; k++)
		{
			eOut[k] = (r.p[k] - (k == labels[i] ? 1 : 0)) / count;

			for (int j = 0; j < HIDDEN; j++)
			{
				g.output[k].w[j] += eOut[k] * r.h[j];
			}
			g.output[k].b += eOut[k];
		}
		
		for (int j = 0; j < HIDDEN; j++)
		{
			double sum = 0;
			for (int k = 0; k < OUTPUTS; k++)
			{
				sum += eOut[k] * m.output[k].w[j];
			}

			double eHid = sum * r.h[j] * (1 - r.h[j]);

			for (int pix = 0; pix < INPUTS; pix++)
			{
				g.hidden[j].w[pix] += eHid * images[i * INPUTS + pix];
			}
			g.hidden[j].b += eHid;
		}
	}

	return g;
}

void updateModel(Model& m, const Model& g, double lr)
{
	for (int i = 0; i < HIDDEN; i++)
	{
		for (int j = 0; j < INPUTS; j++)
		{
			m.hidden[i].w[j] -= lr * g.hidden[i].w[j];
		}
		m.hidden[i].b -= lr * g.hidden[i].b;
	}

	for (int i = 0; i < OUTPUTS; i++)
	{
		for (int j = 0; j < HIDDEN; j++)
		{
			m.output[i].w[j] -= lr * g.output[i].w[j];
		}
		m.output[i].b -= lr * g.output[i].b;
	}
}

void train(Model& m, const std::vector<double>& images, const std::vector<int>& labels, int epochs, double lr)
{
	std::vector<int> order;

	order.reserve(labels.size());

	for (size_t i = 0; i < labels.size(); i++)
	{
		order.push_back(i);
	}

	for (int e = 0; e < epochs; e++)
	{
		std::shuffle(order.begin(), order.end(), rng);

		for (size_t start = 0; start < labels.size(); start += BATCH)
		{
			int count = std::min(BATCH, (int)labels.size() - (int)start);

			Model g = computeGradient(m, images, labels, start, count, order);

			updateModel(m, g, lr);
		}
	}
}

double accuracy(const Model& m, const std::vector<double>& images, const std::vector<int>& labels)
{
	int correct = 0;
	ForwardResult r;

	for (size_t i = 0; i < labels.size(); i++)
	{
		forward(m, &images[i * INPUTS], r);

		int pred = 0;
		for (int k = 1; k < OUTPUTS; k++)
		{
			if (r.p[k] > r.p[pred])
			{
				pred = k;
			}
		}

		if (pred == labels[i])
		{
			correct++;
		}
	}

	return 100.0 * correct / labels.size();
}

int main()
{
	uint32_t trainCount, trainLabelCount, testCount, testLabelCount, rows, cols;
	std::vector<double> trainImages, testImages;
	std::vector<int> trainLabels, testLabels;

	try
	{
		trainImages = loadImages("train-images-idx3-ubyte", trainCount, rows, cols);
		testImages = loadImages("t10k-images-idx3-ubyte", testCount, rows, cols);
		trainLabels = loadLabels("train-labels-idx1-ubyte", trainLabelCount);
		testLabels = loadLabels("t10k-labels-idx1-ubyte", testLabelCount);

		if (trainCount != trainLabelCount)
		{
			throw std::runtime_error("Train files don't correspond");
		}

		if (testCount != testLabelCount)
		{
			throw std::runtime_error("Test files don't correspond");
		}
	}
	catch (const std::exception& e)
	{
		std::cout << "Error: " << e.what() << "\n";
		return 1;
	}

	Model model;

	initModel(model);

	train(model, trainImages, trainLabels, 10, 0.5);

	std::cout << accuracy(model, testImages, testLabels);
}