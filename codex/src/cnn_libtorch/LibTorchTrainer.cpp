#include "LibTorchTrainer.h"
#include "GTSRBDataset.h"

#include <torch/torch.h>
#include <torch/nn/functional/vision.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace cppcnn {

namespace {

float batchAcc(const torch::Tensor& out, const torch::Tensor& tgt) {
    return out.argmax(1).eq(tgt).sum().template item<int64_t>() / static_cast<float>(tgt.size(0));
}

std::pair<float, float> perClassStats(const torch::Tensor& out, const torch::Tensor& tgt, int64_t nc) {
    auto pred = out.argmax(1);
    std::vector<int64_t> cor(static_cast<size_t>(nc), 0), tot(static_cast<size_t>(nc), 0);
    auto pa = pred.accessor<int64_t,1>(), ta = tgt.accessor<int64_t,1>();
    for (int64_t i = 0; i < tgt.size(0); ++i) { int64_t t = ta[i];
        if (t >= 0 && t < nc) { tot[t]++; if (pa[i] == t) cor[t]++; } }
    float mean=0, min=1; int64_t v=0;
    for (int64_t c=0; c<nc; ++c) { auto u=static_cast<size_t>(c);
        if (tot[u]>0) { float a=static_cast<float>(cor[u])/static_cast<float>(tot[u]); mean+=a; min=std::min(min,a); v++; } }
    if (v>0) mean/=static_cast<float>(v); if (v==0) min=0;
    return {mean, min};
}

std::vector<size_t> balancedIdx(const std::vector<TrainSample>& samples, int64_t nc, std::uint32_t seed) {
    std::vector<std::vector<size_t>> bc(static_cast<size_t>(nc));
    for (size_t i=0; i<samples.size(); ++i) { int64_t l=samples[i].label;
        if (l>=0 && l<nc) bc[static_cast<size_t>(l)].push_back(i); }
    size_t mx=0; for (auto& c:bc) mx=std::max(mx,c.size()); if(mx==0) mx=1;
    std::vector<size_t> r; std::mt19937 rg(seed);
    for (auto& c:bc) { if(c.empty()) continue;
        std::uniform_int_distribution<size_t> d(0,c.size()-1);
        for(size_t j=0;j<mx;++j) r.push_back(c[d(rg)]); }
    std::shuffle(r.begin(),r.end(),rg); return r;
}

// ---------------------------------------------------------------------------
// Batch data augmentation using LibTorch ops.
// Images: (B, C, H, W) float tensor in [0, 1].
// Applies: rotation ¡À15 deg, translation ¡À4 px, scaling ¡À10%,
// brightness ¡À0.2, contrast factor 0.8-1.2, Gaussian noise ¦Ò=0.05.
// ---------------------------------------------------------------------------
torch::Tensor augmentBatch(torch::Tensor images, std::mt19937& rng) {
    auto B = images.size(0);
    auto C = images.size(1);
    auto H = images.size(2);
    auto W = images.size(3);
    auto device = images.device();

    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

    // --- Geometric transforms (per-sample affine) ---
    std::vector<torch::Tensor> warped(static_cast<size_t>(B));
    namespace F = torch::nn::functional;
    for (int64_t i = 0; i < B; ++i) {
        float angle = ud(rng) * 15.0f * 3.14159265f / 180.0f;
        float tx = ud(rng) * 4.0f / static_cast<float>(W);
        float ty = ud(rng) * 4.0f / static_cast<float>(H);
        float scale = 1.0f + ud(rng) * 0.1f;
        float c = std::cos(angle) * scale;
        float s = std::sin(angle) * scale;
        auto theta = torch::tensor({{c, -s, tx}, {s, c, ty}}, torch::kFloat32)
            .unsqueeze(0).to(device);
        auto grid = F::affine_grid(theta, torch::IntArrayRef({1, C, H, W}), false);
        warped[static_cast<size_t>(i)] = F::grid_sample(
            images[i].unsqueeze(0), grid,
            F::GridSampleFuncOptions()
                .mode(torch::kBilinear)
                .padding_mode(torch::kZeros)
                .align_corners(false)
        ).squeeze(0);
    }
    auto result = torch::stack(warped);

    // --- Photometric transforms (batch) ---
    // Brightness ¡À0.2 per sample
    auto bright = torch::zeros({B, 1, 1, 1}, torch::kFloat32).to(device);
    for (int64_t i = 0; i < B; ++i) bright[i][0][0][0] = ud(rng) * 0.2f;
    result = result + bright;

    // Contrast factor 0.8-1.2 per sample
    auto means = result.mean({1, 2, 3}, true);
    auto cFactor = torch::ones({B, 1, 1, 1}, torch::kFloat32).to(device);
    for (int64_t i = 0; i < B; ++i) cFactor[i][0][0][0] = 1.0f + ud(rng) * 0.2f;
    result = means + cFactor * (result - means);

    // Gaussian noise sigma = 0.05
    result = result + torch::randn_like(result) * 0.05f;

    return torch::clamp(result, 0.0f, 1.0f);
}

} // anon namespace

LibTorchTrainingReport trainLibTorchModel(
    LibTorchModule& model, torch::Device device,
    const std::vector<TrainSample>& trainSamples,
    const std::vector<TrainSample>& valSamples,
    int64_t numClasses, const LibTorchTrainingOptions& opts)
{
    LibTorchTrainingReport report; report.totalParams=0;
    for (const auto& p : model.parameters()) report.totalParams+=static_cast<std::size_t>(p.numel());

    std::vector<SamplePair> trainPairs, valPairs;
    trainPairs.reserve(trainSamples.size()); valPairs.reserve(valSamples.size());
    for (const auto& s : trainSamples) trainPairs.emplace_back(s.imagePath, s.label);
    for (const auto& s : valSamples) valPairs.emplace_back(s.imagePath, s.label);

    auto trainDS = GTSRBDataset(trainPairs, 32, 32);
    auto valDS = GTSRBDataset(valPairs, 32, 32);

    auto params = model.parameters();
    torch::optim::SGD opt(params, torch::optim::SGDOptions(opts.learningRate).momentum(opts.momentum).weight_decay(opts.weightDecay));
    double currentLr = opts.learningRate;
    auto crit = torch::nn::CrossEntropyLoss();
    std::mt19937 rng(opts.seed);

    std::vector<int64_t> classCor(static_cast<size_t>(numClasses),0), classTot(static_cast<size_t>(numClasses),0);
    auto startTime = std::chrono::steady_clock::now();

    for (size_t ep=0; ep<opts.epochs; ++ep) {
        auto epStart = std::chrono::steady_clock::now();
        model.train();
        float trainLoss=0, trainAcc=0; int64_t trainBatches=0;

        const std::vector<size_t>* idxPtr;
        std::vector<size_t> idxVec;
        if (opts.enableClassBalancing) { idxVec = balancedIdx(trainSamples,numClasses,opts.seed+static_cast<std::uint32_t>(ep)); idxPtr=&idxVec; }
        else { idxVec.resize(trainSamples.size()); std::iota(idxVec.begin(),idxVec.end(),size_t{0}); std::shuffle(idxVec.begin(),idxVec.end(),rng); idxPtr=&idxVec; }

        for (size_t s=0; s<idxPtr->size(); s+=opts.batchSize) {
            size_t e=std::min(s+opts.batchSize,idxPtr->size()), cnt=e-s;
            std::vector<torch::Tensor> imgs; std::vector<int64_t> labs;
            imgs.reserve(cnt); labs.reserve(cnt);
            for (size_t j=s; j<e; ++j) { size_t ix=(*idxPtr)[j];
                imgs.push_back(trainDS.get(ix).data.to(device)); labs.push_back(trainSamples[ix].label); }
            auto data = torch::stack(imgs);
            if (opts.enableAugmentation) {
                data = augmentBatch(data, rng);
            }
            auto targets = torch::tensor(labs, torch::kInt64).to(device);
            opt.zero_grad();
            auto out = model.forward(data);
            auto loss = crit(out, targets);
            loss.backward(); opt.step();
            trainLoss += loss.item<float>(); trainAcc += batchAcc(out, targets); trainBatches++;
        }
        if (device.is_cuda()) torch::cuda::synchronize();

        // ---- Validation on CPU (workaround for CUDA forward crash) ----
        model.eval();
        float valLoss=0, valAcc=0; int64_t valBatches=0;
        std::fill(classCor.begin(), classCor.end(), 0); std::fill(classTot.begin(), classTot.end(), 0);

        if (!valSamples.empty()) {
            model.to(torch::kCPU);
            for (size_t s=0; s<valPairs.size(); s+=opts.batchSize) {
                size_t e=std::min(s+opts.batchSize,valPairs.size()), cnt=e-s;
                std::vector<torch::Tensor> imgs; std::vector<int64_t> labs;
                imgs.reserve(cnt); labs.reserve(cnt);
                for (size_t j=s; j<e; ++j) {
                    auto ex = valDS.get(j);
                    imgs.push_back(ex.data); labs.push_back(ex.target.item<int64_t>());
                }
                auto data = torch::stack(imgs);
                auto targets = torch::tensor(labs, torch::kInt64);
                auto out = model.forward(data);
                auto loss = crit(out, targets);
                valLoss += loss.item<float>(); valAcc += batchAcc(out, targets); valBatches++;
                auto ps = out.argmax(1); auto pa = ps.accessor<int64_t,1>(); auto ta = targets.accessor<int64_t,1>();
                for (int64_t i=0; i<targets.size(0); ++i) { int64_t t=ta[i];
                    if (t>=0 && t<numClasses) { classTot[static_cast<size_t>(t)]++; if (pa[i]==t) classCor[static_cast<size_t>(t)]++; } }
            }
            model.to(device);
        }

        auto epEnd = std::chrono::steady_clock::now();
        float epSec = std::chrono::duration<float>(epEnd-epStart).count();
        trainLoss/=static_cast<float>(trainBatches); trainAcc/=static_cast<float>(trainBatches);

        float meanClass=0, minClass=1; int64_t valid=0;
        for (int64_t c=0; c<numClasses; ++c) { auto u=static_cast<size_t>(c);
            if (classTot[u]>0) { float a=static_cast<float>(classCor[u])/static_cast<float>(classTot[u]); meanClass+=a; minClass=std::min(minClass,a); valid++; } }
        if (valid>0) meanClass/=static_cast<float>(valid); if (valid==0) minClass=0;
        float avgValLoss=valBatches>0?valLoss/static_cast<float>(valBatches):0, avgValAcc=valBatches>0?valAcc/static_cast<float>(valBatches):0;
        if (avgValAcc>report.bestValAccuracy) {
            report.bestValAccuracy=avgValAcc; report.bestEpoch=ep;
            if (!opts.checkpointPath.empty()) {
                try {
                    model.to(torch::kCPU);
                    model.saveToFile(opts.checkpointPath);
                    if (opts.verbose) std::cout << "  (best model saved to " << opts.checkpointPath << ")" << std::endl;
                    model.to(device);
                } catch (const std::exception& e) {
                    std::cerr << "Checkpoint save failed: " << e.what() << std::endl;
                    model.to(device);
                }
            }
        }

        // StepLR: decay learning rate every lrStepSize epochs
        if (opts.enableLrScheduler && opts.lrStepSize > 0) {
            if ((ep + 1) % opts.lrStepSize == 0) {
                double newLr = std::max(currentLr * opts.lrDecayFactor, opts.minLr);
                if (newLr < currentLr) {
                    for (auto& group : opt.param_groups()) {
                        auto& sgdOpts = static_cast<torch::optim::SGDOptions&>(group.options());
                        sgdOpts.lr(newLr);
                    }
                    currentLr = newLr;
                }
            }
        }

        LibTorchEpochMetrics m;
        m.epoch=ep; m.trainLoss=trainLoss; m.trainAccuracy=trainAcc; m.valLoss=avgValLoss; m.valAccuracy=avgValAcc;
        m.meanClassAccuracy=meanClass; m.minClassAccuracy=minClass; m.durationSec=epSec; m.currentLr=currentLr;
        report.history.push_back(m);
        if (opts.verbose) std::cout<<"Epoch "<<(ep+1)<<"/"<<opts.epochs
            <<" | train loss: "<<std::fixed<<std::setprecision(4)<<trainLoss<<" acc: "<<std::setprecision(2)<<(trainAcc*100)<<"%"
            <<" | val loss: "<<std::setprecision(4)<<avgValLoss<<" acc: "<<std::setprecision(2)<<(avgValAcc*100)<<"%"
            <<" mean: "<<(meanClass*100)<<"% min: "<<(minClass*100)<<"% | "<<std::setprecision(1)<<epSec<<"s"<<std::endl;
    }
    report.totalDurationSec = std::chrono::duration<float>(std::chrono::steady_clock::now()-startTime).count();

    // Write CSV history
    if (!opts.csvPath.empty()) {
        std::ofstream csv(opts.csvPath);
        if (csv.is_open()) {
            csv << "epoch,train_loss,train_acc,val_loss,val_acc,mean_class_acc,min_class_acc,lr,duration_sec" << std::endl;
            for (const auto& m : report.history) {
                csv << (m.epoch + 1) << ","
                    << std::fixed << std::setprecision(6) << m.trainLoss << ","
                    << std::setprecision(6) << m.trainAccuracy << ","
                    << std::setprecision(6) << m.valLoss << ","
                    << std::setprecision(6) << m.valAccuracy << ","
                    << std::setprecision(6) << m.meanClassAccuracy << ","
                    << std::setprecision(6) << m.minClassAccuracy << ","
                    << std::setprecision(8) << m.currentLr << ","
                    << std::setprecision(2) << m.durationSec << std::endl;
            }
            if (opts.verbose) std::cout << "CSV history saved to " << opts.csvPath << std::endl;
        } else {
            std::cerr << "Failed to open CSV for writing: " << opts.csvPath << std::endl;
        }
    }

    return report;
}

std::tuple<float,float,float,float> evaluateLibTorchModel(
    LibTorchModule& model, torch::Device device,
    const std::vector<TrainSample>& samples, int64_t numClasses, std::size_t batchSize)
{
    std::vector<SamplePair> pairs; pairs.reserve(samples.size());
    for (const auto& s : samples) pairs.emplace_back(s.imagePath, s.label);
    auto ds = GTSRBDataset(pairs, 32, 32);
    model.eval(); model.to(torch::kCPU);
    auto crit = torch::nn::CrossEntropyLoss();
    float totalLoss=0, totalAcc=0; int64_t batches=0;
    std::vector<int64_t> classCor(static_cast<size_t>(numClasses),0), classTot(static_cast<size_t>(numClasses),0);
    for (size_t s=0; s<pairs.size(); s+=batchSize) {
        size_t e=std::min(s+batchSize,pairs.size()), cnt=e-s;
        std::vector<torch::Tensor> imgs; std::vector<int64_t> labs;
        imgs.reserve(cnt); labs.reserve(cnt);
        for (size_t j=s; j<e; ++j) { auto ex=ds.get(j); imgs.push_back(ex.data); labs.push_back(ex.target.item<int64_t>()); }
        auto data = torch::stack(imgs);
        auto targets = torch::tensor(labs, torch::kInt64);
        auto out = model.forward(data);
        auto loss = crit(out, targets);
        totalLoss+=loss.item<float>(); totalAcc+=batchAcc(out,targets); batches++;
        auto ps=out.argmax(1); auto pa=ps.accessor<int64_t,1>(); auto ta=targets.accessor<int64_t,1>();
        for (int64_t i=0;i<targets.size(0);++i) { int64_t t=ta[i];
            if (t>=0&&t<numClasses) { classTot[static_cast<size_t>(t)]++; if(pa[i]==t) classCor[static_cast<size_t>(t)]++; } }
    }
    float avgLoss=batches>0?totalLoss/static_cast<float>(batches):0, avgAcc=batches>0?totalAcc/static_cast<float>(batches):0;
    float meanClass=0, minClass=1; int64_t valid=0;
    for (int64_t c=0;c<numClasses;++c) { auto u=static_cast<size_t>(c);
        if (classTot[u]>0) { float a=static_cast<float>(classCor[u])/static_cast<float>(classTot[u]); meanClass+=a; minClass=std::min(minClass,a); valid++; } }
    if (valid>0) meanClass/=static_cast<float>(valid); if (valid==0) minClass=0;
    return {avgLoss,avgAcc,meanClass,minClass};
}

} // namespace cppcnn

