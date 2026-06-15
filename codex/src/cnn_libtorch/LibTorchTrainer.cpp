#include "LibTorchTrainer.h"
#include "GTSRBDataset.h"

#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
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
            // Move model to CPU for validation
            model.to(torch::kCPU);
            for (size_t s=0; s<valPairs.size(); s+=opts.batchSize) {
                size_t e=std::min(s+opts.batchSize,valPairs.size()), cnt=e-s;
                std::vector<torch::Tensor> imgs; std::vector<int64_t> labs;
                imgs.reserve(cnt); labs.reserve(cnt);
                for (size_t j=s; j<e; ++j) {
                    auto ex = valDS.get(j);
                    imgs.push_back(ex.data); labs.push_back(ex.target.item<int64_t>());  // keep on CPU
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
            model.to(device);  // move back to CUDA for next epoch
        }

        auto epEnd = std::chrono::steady_clock::now();
        float epSec = std::chrono::duration<float>(epEnd-epStart).count();
        trainLoss/=static_cast<float>(trainBatches); trainAcc/=static_cast<float>(trainBatches);

        float meanClass=0, minClass=1; int64_t valid=0;
        for (int64_t c=0; c<numClasses; ++c) { auto u=static_cast<size_t>(c);
            if (classTot[u]>0) { float a=static_cast<float>(classCor[u])/static_cast<float>(classTot[u]); meanClass+=a; minClass=std::min(minClass,a); valid++; } }
        if (valid>0) meanClass/=static_cast<float>(valid); if (valid==0) minClass=0;
        float avgValLoss=valBatches>0?valLoss/static_cast<float>(valBatches):0, avgValAcc=valBatches>0?valAcc/static_cast<float>(valBatches):0;
        if (avgValAcc>report.bestValAccuracy) { report.bestValAccuracy=avgValAcc; report.bestEpoch=ep; }

        LibTorchEpochMetrics m;
        m.epoch=ep; m.trainLoss=trainLoss; m.trainAccuracy=trainAcc; m.valLoss=avgValLoss; m.valAccuracy=avgValAcc;
        m.meanClassAccuracy=meanClass; m.minClassAccuracy=minClass; m.durationSec=epSec; m.currentLr=opts.learningRate;
        report.history.push_back(m);
        if (opts.verbose) std::cout<<"Epoch "<<(ep+1)<<"/"<<opts.epochs
            <<" | train loss: "<<std::fixed<<std::setprecision(4)<<trainLoss<<" acc: "<<std::setprecision(2)<<(trainAcc*100)<<"%"
            <<" | val loss: "<<std::setprecision(4)<<avgValLoss<<" acc: "<<std::setprecision(2)<<(avgValAcc*100)<<"%"
            <<" mean: "<<(meanClass*100)<<"% min: "<<(minClass*100)<<"% | "<<std::setprecision(1)<<epSec<<"s"<<std::endl;
    }
    report.totalDurationSec = std::chrono::duration<float>(std::chrono::steady_clock::now()-startTime).count();
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

