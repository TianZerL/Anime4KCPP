#include <doctest/doctest.h>

#include "AC/Core.hpp"

template<typename Model>
static void checkBaseModel(Model& model)
{
    CHECK(model.blocks() > 0);
    CHECK(model.kernels() > 0);
    CHECK(model.biases() > 0);

    CHECK(model.kernel(0) != nullptr);
    CHECK(model.bias(0) != nullptr);

    CHECK(model.kernelOffset(0) == 0);

    int lastKernel = model.kernels() - 1;
    CHECK(model.kernelOffset(lastKernel) + model.kernelLength(lastKernel) == model.kernelLength());

    CHECK(model.biasOffset(0) == 0);

    int lastBias = model.biases() - 1;
    CHECK(model.biasOffset(lastBias) + model.biasLength(lastBias) == model.biasLength());

    int kernelSum = 0;
    for (int i = 0; i < model.kernels(); i++)
        kernelSum += model.kernelLength(i);
    CHECK(kernelSum == model.kernelLength());

    int biasSum = 0;
    for (int i = 0; i < model.biases(); i++)
        biasSum += model.biasLength(i);
    CHECK(biasSum == model.biasLength());
}

TEST_CASE("ACNetLegacy HDN0")
{
    ac::core::model::ACNetLegacy model{ ac::core::model::ACNetLegacy::Variant::HDN0 };
    checkBaseModel(model);
    CHECK(model.alphas() == 0);
    CHECK(model.kernels() > 0);
}

TEST_CASE("ACNet F8 B4")
{
    ac::core::model::ACNet<8> model{ ac::core::model::ACNet<8>::Variant::B4_NORMAL };
    checkBaseModel(model);
    CHECK(model.alphas() > 0);

    CHECK(model.alpha(0) != nullptr);
    CHECK(model.alphaOffset(0) == 0);
}

TEST_CASE("ARNet F8 B8")
{
    ac::core::model::ARNet<8> model{ ac::core::model::ARNet<8>::Variant::B8_NORMAL };
    checkBaseModel(model);
    CHECK(model.alphas() > 0);
    CHECK(model.alpha(0) != nullptr);
}

TEST_CASE("ArtCNN C4 F16")
{
    ac::core::model::ArtCNN<16> model{ ac::core::model::ArtCNN<16>::Variant::C4_NORMAL };
    checkBaseModel(model);
}

TEST_CASE("FSRCNNX F8 B4")
{
    ac::core::model::FSRCNNX<8> model{ ac::core::model::FSRCNNX<8>::Variant::B4_NORMAL };
    checkBaseModel(model);
    CHECK(model.alphas() > 0);
}
