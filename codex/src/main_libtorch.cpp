/// cppcnn_app_gpu — LibTorch GPU-accelerated CLI for GTSRB training.
///
/// Usage:
///   train <dataset_dir> <model_out.pt> [options]
///   evaluate <dataset_dir> <model.pt> [options]
///   predict <image> <model.pt> [options]
///
/// Options:
///   --classes N        Number of classes (default: 43)
///   --arch TYPE        LeNet | Enhanced (default: Enhanced)
///   --epochs N         Training epochs (default: 10)
///   --batch N          Batch size (default: 64)
///   --lr F             Learning rate (default: 0.01)
///   --wd F             Weight decay (default: 0.0001)
///   --momentum F       Momentum (default: 0.9)
///   --val F            Validation ratio (default: 0.2)
///   --seed N           Random seed (default: 42)
///   --aug              Enable data augmentation
///   --balance          Enable class balancing
///   --no-cuda          Disable CUDA (force CPU)
///   --verbose          Print detailed progress
///   --help             Show this message

#include "cnn_libtorch/LibTorchModel.h"
#include "cnn_libtorch/LibTorchTrainer.h"
#include "data/DataLoader.h"
#include "image/ImageProcessor.h"

#include <torch/torch.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// ============================================================================
// Argument parsing
// ============================================================================

struct Args {
    std::string command;
    std::vector<std::string> positional;
    std::map<std::string, std::string> flags;

    [[nodiscard]] bool has(const std::string& key) const {
        return flags.find(key) != flags.end();
    }

    [[nodiscard]] std::string get(const std::string& key,
                                  const std::string& def = "") const {
        auto it = flags.find(key);
        return it != flags.end() ? it->second : def;
    }

    [[nodiscard]] int getInt(const std::string& key, int def) const {
        auto it = flags.find(key);
        return it != flags.end() ? std::stoi(it->second) : def;
    }

    [[nodiscard]] float getFloat(const std::string& key, float def) const {
        auto it = flags.find(key);
        return it != flags.end() ? std::stof(it->second) : def;
    }

    [[nodiscard]] double getDouble(const std::string& key, double def) const {
        auto it = flags.find(key);
        return it != flags.end() ? std::stod(it->second) : def;
    }
};

static Args parseArgs(int argc, char* argv[]) {
    Args args;
    if (argc < 2) return args;

    args.command = argv[1];

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.size() >= 2 && arg[0] == '-' && arg[1] == '-') {
            // Named flag: --key=value or --key value
            auto eqPos = arg.find('=');
            if (eqPos != std::string::npos) {
                std::string key = arg.substr(2, eqPos - 2);
                std::string value = arg.substr(eqPos + 1);
                args.flags[key] = value;
            } else {
                std::string key = arg.substr(2);
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    args.flags[key] = argv[++i];
                } else {
                    args.flags[key] = "true";
                }
            }
        } else {
            args.positional.push_back(arg);
        }
    }

    return args;
}

static void printUsage() {
    std::cout << "cppcnn_app_gpu — GPU-accelerated GTSRB training\n"
              << "\n"
              << "Usage:\n"
              << "  train <dataset_dir> <model_out.pt> [options]\n"
              << "  evaluate <dataset_dir> <model.pt> [options]\n"
              << "  predict <image> <model.pt> [options]\n"
              << "\n"
              << "Options:\n"
              << "  --classes N        Number of classes (default: 43)\n"
              << "  --arch TYPE        LeNet | Enhanced (default: Enhanced)\n"
              << "  --epochs N         Training epochs (default: 10)\n"
              << "  --batch N          Batch size (default: 64)\n"
              << "  --lr F             Learning rate (default: 0.01)\n"
              << "  --wd F             Weight decay (default: 0.0001)\n"
              << "  --momentum F       Momentum (default: 0.9)\n"
              << "  --val F            Validation ratio (default: 0.2)\n"
              << "  --seed N           Random seed (default: 42)\n"
              << "  --aug              Enable data augmentation\n"
              << "  --balance          Enable class balancing\n"
              << "  --no-cuda          Disable CUDA (force CPU)\n"
              << "  --verbose          Print detailed progress\n"
              << "  --help             Show this message\n";
}

// ============================================================================
// CUDA device setup
// ============================================================================

static torch::Device setupDevice(bool useCuda) {
    if (useCuda && torch::cuda::is_available()) {
        auto device = torch::Device(torch::kCUDA);
        std::cout << "Using CUDA device ("
                  << static_cast<int>(torch::cuda::device_count()) << " GPU(s) available)"
                  << std::endl;
        return device;
    }
    std::cout << "Using CPU device" << std::endl;
    return torch::Device(torch::kCPU);
}

// ============================================================================
// Convert cppcnn::Dataset samples to TrainSample vector
// ============================================================================

static std::vector<cppcnn::TrainSample> toTrainSamples(
    const cppcnn::Dataset& dataset,
    const cppcnn::DataLoader& loader)
{
    std::vector<cppcnn::TrainSample> result;
    result.reserve(dataset.samples.size());
    for (const auto& s : dataset.samples) {
        result.push_back({s.imagePath.string(), static_cast<int64_t>(s.label)});
    }
    return result;
}

// ============================================================================
// Commands
// ============================================================================

static int cmdTrain(const Args& args) {
    if (args.positional.size() < 2) {
        std::cerr << "Usage: train <dataset_dir> <model_out.pt> [options]\n";
        return 1;
    }

    std::filesystem::path datasetDir = args.positional[0];
    std::filesystem::path modelPath  = args.positional[1];

    int numClasses    = args.getInt("classes", 43);
    std::string archStr = args.get("arch", "Enhanced");
    int epochs        = args.getInt("epochs", 10);
    int batchSize     = args.getInt("batch", 64);
    double lr         = args.getDouble("lr", 0.01);
    double wd         = args.getDouble("wd", 0.0001);
    double momentum   = args.getDouble("momentum", 0.9);
    float valRatio    = args.getFloat("val", 0.2F);
    bool useAug       = args.has("aug");
    bool useBalance   = args.has("balance");
    bool useCuda      = !args.has("no-cuda");
    bool verbose      = args.has("verbose");
    int seed          = args.getInt("seed", 42);

    // Determine architecture
    cppcnn::LibTorchArchitecture arch;
    if (archStr == "LeNet" || archStr == "lenet") {
        arch = cppcnn::LibTorchArchitecture::LeNet;
    } else {
        arch = cppcnn::LibTorchArchitecture::Enhanced;
    }

    std::cout << "=== LibTorch GPU Training ===" << "\n"
              << "Dataset:     " << datasetDir << "\n"
              << "Model out:   " << modelPath << "\n"
              << "Arch:        " << archStr << "\n"
              << "Classes:     " << numClasses << "\n"
              << "Epochs:      " << epochs << "\n"
              << "Batch size:  " << batchSize << "\n"
              << "LR:          " << lr << "\n"
              << "Weight Decay:" << wd << "\n"
              << "Momentum:    " << momentum << "\n"
              << "Val ratio:   " << valRatio << "\n"
              << "Augmentation:" << (useAug ? "yes" : "no") << "\n"
              << "Balancing:   " << (useBalance ? "yes" : "no") << "\n"
              << "Seed:        " << seed << "\n"
              << std::endl;

    // Setup device
    torch::Device device = setupDevice(useCuda);

    // Load dataset using original DataLoader
    std::cout << "Scanning dataset..." << std::endl;
    cppcnn::DataLoaderOptions loaderOpts;
    loaderOpts.classLimit = static_cast<std::size_t>(numClasses);
    loaderOpts.samplesPerClass = 0;  // All samples
    loaderOpts.seed = static_cast<std::uint32_t>(seed);

    cppcnn::DataLoader loader(loaderOpts);
    auto fullDataset = loader.loadDataset(datasetDir, cppcnn::DatasetSplit::Training);

    std::cout << "Loaded " << fullDataset.size() << " training samples, "
              << fullDataset.classCount() << " classes" << std::endl;

    if (fullDataset.empty()) {
        std::cerr << "Error: dataset is empty at " << datasetDir << std::endl;
        return 1;
    }

    // Track-aware validation split
    auto split = cppcnn::DataLoader::splitByTrack(
        fullDataset, valRatio, static_cast<std::uint32_t>(seed));

    auto trainSamples = toTrainSamples(split.train, loader);
    auto valSamples   = toTrainSamples(split.validation, loader);

    std::cout << "Train samples: " << trainSamples.size()
              << " | Val samples: " << valSamples.size() << std::endl;

    // Create model
    std::cout << "Creating " << archStr << " model (" << numClasses << " classes)..."
              << std::endl;

    torch::manual_seed(static_cast<int64_t>(seed));

    // We use a shared_ptr to manage the model's lifetime across AnyModule
    std::shared_ptr<cppcnn::LibTorchModule> model;
    if (arch == cppcnn::LibTorchArchitecture::LeNet) {
        model = std::make_shared<cppcnn::LibTorchLeNetImpl>(numClasses);
    } else {
        model = std::make_shared<cppcnn::LibTorchEnhancedImpl>(numClasses);
    }

    model->to(device);

    std::size_t nParams = cppcnn::countParams(*model);
    std::cout << "Parameters: " << nParams << std::endl;

    // Train
    cppcnn::LibTorchTrainingOptions opts;
    opts.epochs = static_cast<std::size_t>(epochs);
    opts.batchSize = static_cast<std::size_t>(batchSize);
    opts.learningRate = lr;
    opts.weightDecay = wd;
    opts.momentum = momentum;
    opts.validationRatio = valRatio;
    opts.enableAugmentation = useAug;
    opts.enableClassBalancing = useBalance;
    opts.checkpointPath = args.get("checkpoint", "");
    opts.csvPath = args.get("csv", "");
    opts.useCuda = useCuda;
    opts.verbose = true;
    opts.seed = static_cast<std::uint32_t>(seed);

    auto report = cppcnn::trainLibTorchModel(
        *model, device, trainSamples, valSamples, numClasses, opts);

    // Save model
    try {
        model->saveToFile(modelPath.string());
        std::cout << "Model saved to " << modelPath << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error saving model: " << e.what() << std::endl;
    }

    // Summary
    const auto& last = report.history.back();
    std::cout << "\n=== Training Complete ===" << "\n"
              << "Best val accuracy: " << (report.bestValAccuracy * 100.0F) << "%"
              << " (epoch " << (report.bestEpoch + 1) << ")\n"
              << "Final val accuracy: " << (last.valAccuracy * 100.0F) << "%\n"
              << "Final train accuracy: " << (last.trainAccuracy * 100.0F) << "%\n"
              << "Total time: " << report.totalDurationSec << "s ("
              << (report.totalDurationSec / 60.0) << " min)\n"
              << "Parameters: " << report.totalParams << "\n"
              << std::endl;

    return 0;
}

static int cmdEvaluate(const Args& args) {
    if (args.positional.size() < 2) {
        std::cerr << "Usage: evaluate <dataset_dir> <model.pt> [options]\n";
        return 1;
    }

    std::filesystem::path datasetDir = args.positional[0];
    std::filesystem::path modelPath  = args.positional[1];

    int numClasses    = args.getInt("classes", 43);
    std::string archStr = args.get("arch", "Enhanced");
    int batchSize     = args.getInt("batch", 64);
    bool useCuda      = !args.has("no-cuda");

    cppcnn::LibTorchArchitecture arch;
    if (archStr == "LeNet" || archStr == "lenet") {
        arch = cppcnn::LibTorchArchitecture::LeNet;
    } else {
        arch = cppcnn::LibTorchArchitecture::Enhanced;
    }

    torch::Device device = setupDevice(useCuda);

    // Load model
    std::shared_ptr<cppcnn::LibTorchModule> model;
    if (arch == cppcnn::LibTorchArchitecture::LeNet) {
        auto m = std::make_shared<cppcnn::LibTorchLeNetImpl>(numClasses);
        m->loadFromFile(modelPath.string());
        model = m;
    } else {
        auto m = std::make_shared<cppcnn::LibTorchEnhancedImpl>(numClasses);
        m->loadFromFile(modelPath.string());
        model = m;
    }
    model->to(device);
    model->eval();

    // Load dataset
    cppcnn::DataLoaderOptions loaderOpts;
    loaderOpts.classLimit = static_cast<std::size_t>(numClasses);
    cppcnn::DataLoader loader(loaderOpts);
    auto dataset = loader.loadDataset(datasetDir, cppcnn::DatasetSplit::Test);
    auto samples = toTrainSamples(dataset, loader);

    std::cout << "Evaluating on " << samples.size() << " samples..." << std::endl;

    auto [loss, acc, meanClass, minClass] = cppcnn::evaluateLibTorchModel(
        *model, device, samples, numClasses, static_cast<std::size_t>(batchSize));

    std::cout << "\n=== Evaluation Results ===" << "\n"
              << "Loss:       " << std::fixed << std::setprecision(4) << loss << "\n"
              << "Accuracy:   " << std::setprecision(2) << (acc * 100.0F) << "%\n"
              << "Mean class: " << (meanClass * 100.0F) << "%\n"
              << "Min class:  " << (minClass * 100.0F) << "%\n"
              << std::endl;

    return 0;
}

static int cmdPredict(const Args& args) {
    if (args.positional.size() < 2) {
        std::cerr << "Usage: predict <image> <model.pt> [options]\n";
        return 1;
    }

    std::filesystem::path imagePath = args.positional[0];
    std::filesystem::path modelPath = args.positional[1];

    int numClasses    = args.getInt("classes", 43);
    std::string archStr = args.get("arch", "Enhanced");
    bool useCuda      = !args.has("no-cuda");

    cppcnn::LibTorchArchitecture arch;
    if (archStr == "LeNet" || archStr == "lenet") {
        arch = cppcnn::LibTorchArchitecture::LeNet;
    } else {
        arch = cppcnn::LibTorchArchitecture::Enhanced;
    }

    torch::Device device = setupDevice(useCuda);

    // Load model
    std::shared_ptr<cppcnn::LibTorchModule> model;
    if (arch == cppcnn::LibTorchArchitecture::LeNet) {
        auto m = std::make_shared<cppcnn::LibTorchLeNetImpl>(numClasses);
        m->loadFromFile(modelPath.string());
        model = m;
    } else {
        auto m = std::make_shared<cppcnn::LibTorchEnhancedImpl>(numClasses);
        m->loadFromFile(modelPath.string());
        model = m;
    }
    model->to(device);
    model->eval();

    // Load and preprocess image
    cppcnn::Tensor cpuTensor = cppcnn::ImageProcessor::loadAndPreprocess(imagePath, 32, 32);
    const auto& data = cpuTensor.values();
    torch::Tensor img = torch::from_blob(
        const_cast<float*>(data.data()),
        {1, 3, 32, 32},
        torch::kFloat32
    ).clone().to(device);

    // Predict
    torch::NoGradGuard noGrad;
    auto output = model->forward(img);  // log-probs
    auto probs = torch::exp(output);    // convert to probabilities
    auto topProb = probs.max(1);
    int64_t predClass = std::get<1>(topProb).item<int64_t>();
    float confidence = std::get<0>(topProb).item<float>() * 100.0F;

    std::cout << "Predicted class: " << predClass
              << " (confidence: " << std::fixed << std::setprecision(2)
              << confidence << "%)" << std::endl;

    return 0;
}

// ============================================================================
// main
// ============================================================================

int cmdCudaTest() {
    using namespace cppcnn;
    std::cout << "=== CUDA Self-Test ===" << std::endl;
    
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA not available!" << std::endl;
        return 1;
    }
    
    auto device = torch::Device(torch::kCUDA);
    
    try {
        // Step 1: Create LeNet model
        std::cout << "1. Creating LeNet model..." << std::endl;
        auto model = std::make_shared<LibTorchLeNetImpl>(10);
        model->to(device);
        std::cout << "2. Model on CUDA" << std::endl;
        
        // Step 2: Forward pass
        std::cout << "3. Creating input..." << std::endl;
        auto input = torch::randn({2, 3, 32, 32}, device);
        std::cout << "4. Forward pass..." << std::endl;
        auto output = model->forward(input);
        std::cout << "5. Output shape: " << output.sizes() << std::endl;
        
        // Step 3: Backward
        std::cout << "6. Creating target..." << std::endl;
        auto target = torch::tensor({0, 1}, torch::kLong).to(device);
        auto criterion = torch::nn::CrossEntropyLoss();
        auto loss = criterion(output, target);
        std::cout << "7. Loss: " << loss.item<float>() << std::endl;
        
        std::cout << "8. Backward..." << std::endl;
        loss.backward();
        
        // Step 4: Optimizer step
        std::cout << "9. Creating SGD optimizer..." << std::endl;
        auto params = model->parameters();
        torch::optim::SGD optimizer(params, torch::optim::SGDOptions(0.01).momentum(0.9).weight_decay(0.0001));
        std::cout << "10. Optimizer step..." << std::endl;
        optimizer.step();
        
        torch::cuda::synchronize();
        std::cout << "11. ALL CUDA TESTS PASSED!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "CUDA TEST FAILED: " << e.what() << std::endl;
        return 1;
    }
}

int main(int argc, char* argv[]) {
    try {
        auto args = parseArgs(argc, argv);

        if (args.command.empty() || args.command == "help" || args.command == "--help") {
            printUsage();
            return 0;
        }

        if (args.command == "train") {
            return cmdTrain(args);
        } else if (args.command == "evaluate") {
            return cmdEvaluate(args);
        } else if (args.command == "cuda-test") {
            return cmdCudaTest();
        } else if (args.command == "predict") {
            return cmdPredict(args);
        } else {
            std::cerr << "Unknown command: " << args.command << "\n";
            printUsage();
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}







