#include <iterator>
#include <tuple>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "AC/Core.hpp"
#include "AC/Specs.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pyac, m)
{
    m.doc() = "Anime4KCPP: A high performance anime upscaler.";

    auto core = m.def_submodule("core");

    py::class_<ac::core::Processor, std::shared_ptr<ac::core::Processor>>(core, "Processor")
        .def(py::init([](const int processorType, const int device, const char* model) {
            return ac::core::Processor::create(processorType, device, model);
        }), py::arg("processor_type") = ac::core::Processor::CPU, py::arg("device") = 0, py::arg("model") = ac::specs::ModelNameList[0])
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
        .def("ok", &ac::core::Processor::ok)
        .def("error", &ac::core::Processor::error)
        .def("name", &ac::core::Processor::name)
        .def_property_readonly_static("InfoList", [](py::object /* self */) {
            return std::tuple{
                ac::core::Processor::info<ac::core::Processor::CPU>(),
#           ifdef AC_CORE_WITH_OPENCL
                ac::core::Processor::info<ac::core::Processor::OpenCL>(),
#           endif
#           ifdef AC_CORE_WITH_CUDA
                ac::core::Processor::info<ac::core::Processor::CUDA>(),
#           endif
            };
        })
        .def_readonly_static("CPU", &ac::core::Processor::CPU)
        .def_readonly_static("OpenCL", &ac::core::Processor::OpenCL)
        .def_readonly_static("CUDA", &ac::core::Processor::CUDA);

    py::enum_<ac::core::ImreadModes>(core, "ImreadModes")
        .value("IMREAD_UNCHANGED", ac::core::IMREAD_UNCHANGED)
        .value("IMREAD_GRAYSCALE", ac::core::IMREAD_GRAYSCALE)
        .value("IMREAD_COLOR", ac::core::IMREAD_COLOR)
        .value("IMREAD_RGB", ac::core::IMREAD_RGB)
        .value("IMREAD_RGBA", ac::core::IMREAD_RGBA)
        .export_values();

    core.def("imread", [](const char* filename, ac::core::ImreadModes flag) {
        auto img = ac::core::imread(filename, flag);
        return py::array_t<std::uint8_t>{
            { img.height(), img.width(), img.channels() },
            { img.stride(), img.pixelSize(), img.elementSize() },
            img.data(),
            py::capsule{ new ac::core::Image{ img }, [](void* v) { delete static_cast<ac::core::Image*>(v); } }
        };
    }, py::arg("filename"), py::arg("flag"));

    core.def("imwrite", [](const char* filename, const py::array_t<std::uint8_t> in) {
        auto src = in.request();
        if (src.ndim != 3) throw py::buffer_error{ "Incompatible dimension: expected 3." };

        return ac::core::imwrite(filename, { static_cast<int>(src.shape[1]), static_cast<int>(src.shape[0]), static_cast<int>(src.shape[2]), ac::core::Image::UInt8, src.ptr, static_cast<int>(src.strides[0]) });
    }, py::arg("filename"), py::arg("image"));

    auto specs = m.def_submodule("specs");

    auto makeTuple = [](auto& arr) {
        auto size = std::size(arr);
        py::tuple tuple{ size };
        for (decltype(size) i = 0; i < size; i++) tuple[i] = arr[i];
        return tuple;
    };

    specs.attr("ModelNameList") = makeTuple(ac::specs::ModelNameList);
    specs.attr("ModelDescriptionList") = makeTuple(ac::specs::ModelDescriptionList);
    specs.attr("ProcessorNameList") = makeTuple(ac::specs::ProcessorNameList);
    specs.attr("ProcessorDescriptionList") = makeTuple(ac::specs::ProcessorDescriptionList);
}
