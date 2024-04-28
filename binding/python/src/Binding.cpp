#include <tuple>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "AC/Core.hpp"

namespace py = pybind11;

struct ModelType
{
    static constexpr int ACNetHDN0 = 0;
    static constexpr int ACNetHDN1 = 1;
    static constexpr int ACNetHDN2 = 2;
    static constexpr int ACNetHDN3 = 3;
};

PYBIND11_MODULE(pyac, m)
{
    m.doc() = "Anime4KCPP: A high performance anime upscaler.";

    auto core = m.def_submodule("core");

    core.attr("ACNET_HDN0") = ModelType::ACNetHDN0;
    core.attr("ACNET_HDN1") = ModelType::ACNetHDN1;
    core.attr("ACNET_HDN2") = ModelType::ACNetHDN2;
    core.attr("ACNET_HDN3") = ModelType::ACNetHDN3;

    py::class_<ac::core::Processor, std::shared_ptr<ac::core::Processor>>(core, "Processor")
        .def(py::init([](const int processorType, const int device, const int modelType) {
            ac::core::model::ACNet model{ [&](){
                    switch (modelType)
                    {
                    case ModelType::ACNetHDN0 : return ac::core::model::ACNet::Variant::HDN0;
                    case ModelType::ACNetHDN1 : return ac::core::model::ACNet::Variant::HDN1;
                    case ModelType::ACNetHDN2 : return ac::core::model::ACNet::Variant::HDN2;
                    case ModelType::ACNetHDN3 : return ac::core::model::ACNet::Variant::HDN3;
                    default: throw py::value_error{ "unknown model type" };
                    }
                }()
            };
            switch (processorType)
            {
            case ac::core::Processor::CPU : return ac::core::Processor::create<ac::core::Processor::CPU>(device, model);
#           ifdef AC_CORE_WITH_OPENCL
                case ac::core::Processor::OpenCL : return ac::core::Processor::create<ac::core::Processor::OpenCL>(device, model);
#           endif
#           ifdef AC_CORE_WITH_CUDA
                case ac::core::Processor::CUDA : return ac::core::Processor::create<ac::core::Processor::CUDA>(device, model);
#           endif
            default: throw py::value_error{ "unsupported processor" };
            }
        }), py::arg("processor_type") = ac::core::Processor::CPU, py::arg("device") = 0, py::arg("model_type") = ModelType::ACNetHDN0)
        .def("process", [](ac::core::Processor& self, const py::array in, const double factor) {
            auto src = in.request();
            if (src.ndim != 3) throw py::buffer_error{ "Incompatible dimension: expected 3." };

            auto type = [&]() -> int {
                if (src.format == py::format_descriptor<std::uint8_t>::format()) return ac::core::Image::UInt8;
                if (src.format == py::format_descriptor<std::uint16_t>::format()) return ac::core::Image::UInt16;
                if (src.format == py::format_descriptor<float>::format()) return ac::core::Image::Float32;
                throw py::buffer_error{ "Incompatible type: expected uint8, uint16 or float32." };
            }();
            py::array out{ in.dtype(), { static_cast<py::ssize_t>(src.shape[0] * factor), static_cast<py::ssize_t>(src.shape[1] * factor), src.shape[2] } };
            auto dst = out.request();
            ac::core::Image srci{ static_cast<int>(src.shape[1]), static_cast<int>(src.shape[0]), static_cast<int>(src.shape[2]), type, src.ptr, static_cast<int>(src.strides[0]) };
            ac::core::Image dsti{ static_cast<int>(dst.shape[1]), static_cast<int>(dst.shape[0]), static_cast<int>(dst.shape[2]), type, dst.ptr, static_cast<int>(dst.strides[0]) };
            self.process(srci, dsti, factor);

            return out;
        }, py::arg("src"), py::arg("factor") = 2.0)
        .def("ok", ac::core::Processor::ok)
        .def("error", ac::core::Processor::error)
        .def("name", ac::core::Processor::name)
        .def_static("info_list", []() {
            return std::make_tuple(
                ac::core::Processor::info<ac::core::Processor::CPU>(),
#           ifdef AC_CORE_WITH_OPENCL
                ac::core::Processor::info<ac::core::Processor::OpenCL>(),
#           else
                "OpenCL: unsupported processor",
#           endif
#           ifdef AC_CORE_WITH_CUDA
                ac::core::Processor::info<ac::core::Processor::CUDA>()
#           else
                "CUDA: unsupported processor"
#           endif
            );
        })
        .def_readonly_static("CPU", &ac::core::Processor::CPU)
        .def_readonly_static("OpenCL", &ac::core::Processor::OpenCL)
        .def_readonly_static("CUDA", &ac::core::Processor::CUDA);
}
