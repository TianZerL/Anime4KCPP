#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "AC/Core.hpp"

namespace py = pybind11;

enum ModelType
{
    AC_MODEL_ACNET_HDN0,
    AC_MODEL_ACNET_HDN1,
    AC_MODEL_ACNET_HDN2,
    AC_MODEL_ACNET_HDN3
};

PYBIND11_MODULE(pyac, m)
{
    m.doc() = "Anime4KCPP: A high performance anime upscaler.";

    py::enum_<ModelType>(m, "ModelType")
        .value("ACNET_HDN0", AC_MODEL_ACNET_HDN0)
        .value("ACNET_HDN1", AC_MODEL_ACNET_HDN1)
        .value("ACNET_HDN2", AC_MODEL_ACNET_HDN2)
        .value("ACNET_HDN3", AC_MODEL_ACNET_HDN3)
        .export_values();

    py::class_<ac::core::Processor, std::shared_ptr<ac::core::Processor>>(m, "Processor")
        .def(py::init([&](const int processorType, const int device, const ModelType modelType) {
            ac::core::model::ACNet model{
                [&](){
                    switch (modelType)
                    {
                    case AC_MODEL_ACNET_HDN0 : return ac::core::model::ACNet::Variant::HDN0;
                    case AC_MODEL_ACNET_HDN1 : return ac::core::model::ACNet::Variant::HDN1;
                    case AC_MODEL_ACNET_HDN2 : return ac::core::model::ACNet::Variant::HDN2;
                    case AC_MODEL_ACNET_HDN3 : return ac::core::model::ACNet::Variant::HDN3;
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
        }), py::arg("processor_type") = ac::core::Processor::CPU, py::arg("device") = 0, py::arg("model_type") = AC_MODEL_ACNET_HDN0)
        .def("process", [](ac::core::Processor& self, const py::array in, const double factor) {
            auto src = in.request();
            auto type = [&]() -> int {
                if (src.format == py::format_descriptor<std::uint8_t>::format()) return ac::core::Image::UInt8;
                if (src.format == py::format_descriptor<std::uint16_t>::format()) return ac::core::Image::UInt16;
                if (src.format == py::format_descriptor<float>::format()) return ac::core::Image::Float32;
                return 0;
            }();
            if (!type) throw py::buffer_error{ "Incompatible type: expected uint8, uint16 or float32." };
            if (src.ndim != 3) throw py::buffer_error{ "Incompatible dimension: expected 3." };

            py::array out{ in.dtype(), { static_cast<py::ssize_t>(src.shape[0] * factor), static_cast<py::ssize_t>(src.shape[1] * factor), src.shape[2] } };
            auto dst = out.request();
            ac::core::Image srci{ static_cast<int>(src.shape[1]), static_cast<int>(src.shape[0]), static_cast<int>(src.shape[2]), type, src.ptr, static_cast<int>(src.strides[0]) };
            ac::core::Image dsti{ static_cast<int>(dst.shape[1]), static_cast<int>(dst.shape[0]), static_cast<int>(dst.shape[2]), type, dst.ptr, static_cast<int>(dst.strides[0]) };
            self.process(srci, dsti, factor);

            return out;
        }, py::arg("src"), py::arg("factor"))
        .def("ok", ac::core::Processor::ok)
        .def("error", ac::core::Processor::error)
        .def("name", ac::core::Processor::name)
        .def_static("info", [](const int processorType){
            switch (processorType)
            {
            case ac::core::Processor::CPU : return ac::core::Processor::info<ac::core::Processor::CPU>();
#           ifdef AC_CORE_WITH_OPENCL
                case ac::core::Processor::OpenCL : return ac::core::Processor::info<ac::core::Processor::OpenCL>();
#           endif
#           ifdef AC_CORE_WITH_CUDA
                case ac::core::Processor::CUDA : return ac::core::Processor::info<ac::core::Processor::CUDA>();
#           endif
            default: return "unsupported processor";
            }
        }, py::arg("processor_type"))
        .def_readonly_static("CPU", &ac::core::Processor::CPU)
        .def_readonly_static("OpenCL", &ac::core::Processor::OpenCL)
        .def_readonly_static("CUDA", &ac::core::Processor::CUDA);
}
