#include "app/App.h"

#include "cnn/Trainer.h"
#include "data/DataLoader.h"
#include "image/ImageProcessor.h"
#include "ui/SimpleUI.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace cppcnn {

namespace {

std::filesystem::path labelPathFrom(
    const std::vector<std::string>& arguments,
    const std::size_t index) {
    return arguments.size() > index
        ? std::filesystem::path(arguments[index])
        : std::filesystem::path("assets/labels.txt");
}

void printPrediction(
    CNN& network,
    const std::filesystem::path& imagePath,
    const std::vector<std::string>& labels,
    const bool showWindow) {
    const Tensor input = ImageProcessor::loadAndPreprocess(imagePath, 32, 32);
    float confidence = 0.0F;
    const std::size_t prediction = network.predict(input, &confidence);
    const std::string& label = labels.at(prediction);

    std::cout
        << "Prediction: [" << prediction << "] " << label << '\n'
        << "Confidence: " << std::fixed << std::setprecision(2)
        << confidence * 100.0F << "%\n";
    if (showWindow) {
        SimpleUI::showPrediction(imagePath, label, confidence);
    }
}

}  // namespace

int App::run(const int argc, char* argv[]) {
    if (argc < 2) {
        printResourceStatus();
        std::cout << '\n';
        printUsage();
        return 0;
    }

    std::vector<std::string> arguments;
    arguments.reserve(static_cast<std::size_t>(argc - 2));
    for (int index = 2; index < argc; ++index) {
        arguments.emplace_back(argv[index]);
    }

    const std::string command = argv[1];
    if (command == "train") {
        return runTrain(arguments);
    }
    if (command == "evaluate") {
        return runEvaluate(arguments);
    }
    if (command == "predict") {
        return runPredict(arguments);
    }
    if (command == "interactive") {
        return runInteractive(arguments);
    }
    if (command == "help" || command == "--help" || command == "-h") {
        printUsage();
        return 0;
    }
    throw std::invalid_argument("Unknown command: " + command);
}

int App::runTrain(const std::vector<std::string>& arguments) {
    if (arguments.size() < 2) {
        throw std::invalid_argument(
            "train requires <train_directory> and <model_path>.");
    }

    const std::size_t requestedClassCount =
        arguments.size() > 2 ? parseSize(arguments[2], "class_count") : 10;
    const std::size_t epochs =
        arguments.size() > 3 ? parseSize(arguments[3], "epochs") : 5;
    const std::size_t samplesPerClass =
        arguments.size() > 4 ? parseSize(arguments[4], "samples_per_class", true) : 0;
    requireDataset(arguments[0], "training");

    DataLoaderOptions loaderOptions;
    loaderOptions.classLimit = requestedClassCount;
    loaderOptions.samplesPerClass = samplesPerClass;
    loaderOptions.shuffle = true;
    DataLoader loader(loaderOptions);
    const Dataset trainingSet =
        loader.loadDataset(arguments[0], DatasetSplit::Training);

    CNN network(trainingSet.classCount());
    TrainingOptions trainingOptions;
    trainingOptions.epochs = epochs;

    std::cout
        << "Training samples: " << trainingSet.size() << '\n'
        << "Classes: " << trainingSet.classCount() << '\n';
    Trainer::train(
        network,
        trainingSet,
        loader,
        trainingOptions,
        &std::cout);
    network.saveModel(arguments[1]);
    std::cout << "Model saved to: " << arguments[1] << '\n';
    return 0;
}

int App::runEvaluate(const std::vector<std::string>& arguments) {
    if (arguments.size() < 2) {
        throw std::invalid_argument(
            "evaluate requires <test_directory> and <model_path>.");
    }
    requireDataset(arguments[0], "evaluation");
    requireModel(arguments[1]);

    const std::size_t classCount = CNN::modelClassCount(arguments[1]);
    const std::size_t samplesPerClass =
        arguments.size() > 2 ? parseSize(arguments[2], "samples_per_class", true) : 0;

    DataLoaderOptions loaderOptions;
    loaderOptions.classLimit = classCount;
    loaderOptions.samplesPerClass = samplesPerClass;
    DataLoader loader(loaderOptions);
    const Dataset testSet =
        loader.loadDataset(arguments[0], DatasetSplit::Test);

    CNN network(classCount);
    network.loadModel(arguments[1]);
    const EpochMetrics metrics = Trainer::evaluate(network, testSet, loader);
    std::cout
        << "Evaluation samples: " << metrics.samples << '\n'
        << "Average loss: " << std::fixed << std::setprecision(4)
        << metrics.averageLoss << '\n'
        << "Accuracy: " << std::setprecision(2)
        << metrics.accuracy * 100.0F << "%\n";
    return 0;
}

int App::runPredict(const std::vector<std::string>& arguments) {
    if (arguments.size() < 2) {
        throw std::invalid_argument("predict requires <image_path> and <model_path>.");
    }
    if (!std::filesystem::is_regular_file(arguments[0])) {
        throw std::runtime_error(
            "Prediction image does not exist: " + arguments[0]
            + ". Choose an image from datasets/GTSRB_subset/test or the full GTSRB test set.");
    }
    requireModel(arguments[1]);

    const std::size_t classCount = CNN::modelClassCount(arguments[1]);
    CNN network(classCount);
    network.loadModel(arguments[1]);
    const auto labels = loadLabels(labelPathFrom(arguments, 2), classCount);
    printPrediction(network, arguments[0], labels, true);
    return 0;
}

int App::runInteractive(const std::vector<std::string>& arguments) {
    if (arguments.empty()) {
        throw std::invalid_argument("interactive requires <model_path>.");
    }
    requireModel(arguments[0]);

    const std::size_t classCount = CNN::modelClassCount(arguments[0]);
    CNN network(classCount);
    network.loadModel(arguments[0]);
    const auto labels = loadLabels(labelPathFrom(arguments, 1), classCount);

    std::cout << "Interactive mode. Enter an image path, or q to quit.\n";
    std::string imagePath;
    while (true) {
        std::cout << "\nImage path> ";
        if (!std::getline(std::cin, imagePath) || imagePath == "q" || imagePath == "quit") {
            break;
        }
        if (imagePath.empty()) {
            continue;
        }
        try {
            printPrediction(network, imagePath, labels, true);
        } catch (const std::exception& error) {
            std::cerr << "Prediction failed: " << error.what() << '\n';
        }
    }
    return 0;
}

void App::printUsage() {
    std::cout
        << "Pure C++ CNN traffic sign recognizer\n\n"
        << "Usage:\n"
        << "  cppcnn_app train <train_dir> <model> [class_count=10]"
           " [epochs=5] [samples_per_class=0]\n"
        << "  cppcnn_app evaluate <test_dir> <model> [samples_per_class=0]\n"
        << "  cppcnn_app predict <image> <model> [labels_file]\n"
        << "  cppcnn_app interactive <model> [labels_file]\n"
        << "  cppcnn_app help\n\n"
        << "A zero samples_per_class value means no per-class limit.\n";
}

void App::printResourceStatus() {
    const std::filesystem::path fullDataset = "datasets/GTSRB";
    const std::filesystem::path subsetDataset = "datasets/GTSRB_subset";
    const std::filesystem::path defaultModel = "models/gtsrb10.bin";

    std::cout
        << "Resource status:\n"
        << "  Full GTSRB: "
        << (std::filesystem::exists(fullDataset) ? "found" : "missing")
        << " (" << fullDataset.string() << ")\n"
        << "  Development subset: "
        << (std::filesystem::exists(subsetDataset) ? "found" : "missing")
        << " (" << subsetDataset.string() << ")\n"
        << "  Default model: "
        << (std::filesystem::is_regular_file(defaultModel) ? "found" : "missing")
        << " (" << defaultModel.string() << ")\n";

    if (!std::filesystem::exists(fullDataset)) {
        std::cout
            << "  Dataset help: read docs/dataset_guide.md and datasets/README.md.\n";
    }
    if (!std::filesystem::is_regular_file(defaultModel)) {
        std::cout
            << "  Model help: train a model first; model binaries are intentionally not in Git.\n";
    }
}

void App::requireDataset(
    const std::filesystem::path& path,
    const std::string& purpose) {
    if (!std::filesystem::is_directory(path)) {
        throw std::runtime_error(
            "Dataset directory for " + purpose + " is missing: " + path.string()
            + ". Download the official GTSRB files into datasets/GTSRB or run "
              "cppcnn_create_subset; see docs/dataset_guide.md.");
    }
}

void App::requireModel(const std::filesystem::path& path) {
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error(
            "Model file is missing: " + path.string()
            + ". Train one with 'cppcnn_app train <dataset> <model>' first. "
              "Model weights are intentionally excluded from Git.");
    }
}

std::size_t App::parseSize(
    const std::string& value,
    const std::string& argumentName,
    const bool allowZero) {
    std::size_t parsedCharacters = 0;
    unsigned long long parsed = 0;
    try {
        parsed = std::stoull(value, &parsedCharacters);
    } catch (const std::exception&) {
        throw std::invalid_argument(argumentName + " must be a non-negative integer.");
    }
    if (parsedCharacters != value.size() || (!allowZero && parsed == 0)) {
        throw std::invalid_argument(
            argumentName + (allowZero ? " must be a non-negative integer." : " must be positive."));
    }
    return static_cast<std::size_t>(parsed);
}

std::vector<std::string> App::loadLabels(
    const std::filesystem::path& path,
    const std::size_t classCount) {
    std::vector<std::string> labels;
    std::ifstream input(path);
    std::string line;
    while (input && std::getline(input, line) && labels.size() < classCount) {
        if (!line.empty()) {
            labels.push_back(line);
        }
    }
    while (labels.size() < classCount) {
        labels.push_back("class " + std::to_string(labels.size()));
    }
    if (!input && !std::filesystem::exists(path)) {
        std::cerr
            << "Warning: labels file not found; using numeric class names: "
            << path.string() << '\n';
    }
    return labels;
}

}  // namespace cppcnn
