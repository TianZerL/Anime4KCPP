#include <cstdint>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <tuple>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "AC/Core.hpp"
#include "AC/Specs.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pyac, m)
{
    m.doc() = "Anime4KCPP: A high performance anime upscaler.";

    auto core = m.def_submodule("core");

    auto processNumpyArray = [](ac::core::Processor& self, const py::array in, const double factor) {
        auto src = in.request();
        if (src.ndim != 2 && src.ndim != 3) throw py::buffer_error{ "Incompatible dimension: expected 2 or 3." };

        int srcH = static_cast<int>(src.shape[0]);
        int srcW = static_cast<int>(src.shape[1]);
        int srcC = src.ndim == 2 ? 1 : static_cast<int>(src.shape[2]);

        int dstH = static_cast<int>(src.shape[0] * factor);
        int dstW = static_cast<int>(src.shape[1] * factor);
        int dstC = srcC;

        auto type = [&]() -> int {
            if (src.format == py::format_descriptor<std::uint8_t>::format()) return ac::core::Image::UInt8;
            if (src.format == py::format_descriptor<std::uint16_t>::format()) return ac::core::Image::UInt16;
            if (src.format == py::format_descriptor<float>::format()) return ac::core::Image::Float32;
            throw py::buffer_error{ "Incompatible type: expected uint8, uint16 or float32." };
            }();
        py::array out{ in.dtype(), dstC == 1 ? py::array::ShapeContainer{ dstH, dstW } : py::array::ShapeContainer{ dstH, dstW, dstC } };
        auto dst = out.request();
        ac::core::Image srci{ srcW, srcH, srcC, type, src.ptr, static_cast<int>(src.strides[0]) };
        ac::core::Image dsti{ dstW, dstH, dstC, type, dst.ptr, static_cast<int>(dst.strides[0]) };
        self.process(srci, dsti, factor);

        return out;
    };

    py::class_<ac::core::Processor, std::shared_ptr<ac::core::Processor>>(core, "Processor")
        .def(py::init([](const char* type, const int device, const char* model) {
            return ac::core::Processor::create(type, device, model);
        }), py::arg("type") = "cpu", py::arg("device") = 0, py::arg("model") = ac::specs::ModelList[0])
        .def("process", processNumpyArray, py::arg("src"), py::arg("factor") = 2.0)
        .def("ok", &ac::core::Processor::ok)
        .def("error", &ac::core::Processor::error)
        .def("name", &ac::core::Processor::name)
        .def("__call__", [&](ac::core::Processor& self, const py::array in, const double factor) {
            auto&& out = processNumpyArray(self, in, factor);
            if (!self.ok()) throw std::runtime_error{ self.error() };
            return out;
        }, py::arg("src"), py::arg("factor") = 2.0)
        .def("__str__", [](ac::core::Processor& self) { return self.name(); })
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

    py::enum_<ac::core::ResizeModes>(core, "ResizeModes")
        .value("RESIZE_POINT", ac::core::RESIZE_POINT)
        .value("RESIZE_CATMULL_ROM", ac::core::RESIZE_CATMULL_ROM)
        .value("RESIZE_MITCHELL_NETRAVALI", ac::core::RESIZE_MITCHELL_NETRAVALI)
        .value("RESIZE_BICUBIC_0_60", ac::core::RESIZE_BICUBIC_0_60)
        .value("RESIZE_BICUBIC_0_75", ac::core::RESIZE_BICUBIC_0_75)
        .value("RESIZE_BICUBIC_0_100", ac::core::RESIZE_BICUBIC_0_100)
        .value("RESIZE_BICUBIC_20_50", ac::core::RESIZE_BICUBIC_20_50)
        .value("RESIZE_SOFTCUBIC50", ac::core::RESIZE_SOFTCUBIC50)
        .value("RESIZE_SOFTCUBIC75", ac::core::RESIZE_SOFTCUBIC75)
        .value("RESIZE_SOFTCUBIC100", ac::core::RESIZE_SOFTCUBIC100)
        .value("RESIZE_LANCZOS2", ac::core::RESIZE_LANCZOS2)
        .value("RESIZE_LANCZOS3", ac::core::RESIZE_LANCZOS3)
        .value("RESIZE_LANCZOS4", ac::core::RESIZE_LANCZOS4)
        .value("RESIZE_SPLINE16", ac::core::RESIZE_SPLINE16)
        .value("RESIZE_SPLINE36", ac::core::RESIZE_SPLINE36)
        .value("RESIZE_SPLINE64", ac::core::RESIZE_SPLINE64)
        .value("RESIZE_BILINEAR", ac::core::RESIZE_BILINEAR)
        .export_values();

    core.def("resize", [](const py::array in, const py::tuple dsize, const double fx, const double fy, ac::core::ResizeModes mode) {
        auto src = in.request();
        if (src.ndim != 2 && src.ndim != 3) throw py::buffer_error{ "Incompatible dimension: expected 2 or 3." };

        int srcH = static_cast<int>(src.shape[0]);
        int srcW = static_cast<int>(src.shape[1]);
        int srcC = src.ndim == 2 ? 1 : static_cast<int>(src.shape[2]);

        int dstH = static_cast<int>(src.shape[0] * fy);
        int dstW = static_cast<int>(src.shape[1] * fx);
        int dstC = srcC;

        if (!dsize.is_none())
        {
            if (dsize.size() != 2) throw py::value_error{ "dsize shouble be (width, height)" };
            dstW = dsize[0].cast<int>();
            dstH = dsize[1].cast<int>();
        }

        auto type = [&]() -> int {
            if (src.format == py::format_descriptor<std::uint8_t>::format()) return ac::core::Image::UInt8;
            if (src.format == py::format_descriptor<std::uint16_t>::format()) return ac::core::Image::UInt16;
            if (src.format == py::format_descriptor<float>::format()) return ac::core::Image::Float32;
            throw py::buffer_error{ "Incompatible type: expected uint8, uint16 or float32." };
        }();

        py::array out{ in.dtype(), dstC == 1 ? py::array::ShapeContainer{ dstH, dstW } : py::array::ShapeContainer{ dstH, dstW, dstC } };
        auto dst = out.request();
        ac::core::Image srci{ srcW, srcH, srcC, type, src.ptr, static_cast<int>(src.strides[0]) };
        ac::core::Image dsti{ dstW, dstH, dstC, type, dst.ptr, static_cast<int>(dst.strides[0]) };
        ac::core::resize(srci, dsti, 0.0, 0.0, mode);

        return out;
    }, py::arg("src"), py::arg("dsize"), py::arg("fx") = 0.0, py::arg("fy") = 0.0, py::arg("mode") = ac::core::RESIZE_BILINEAR);

    py::enum_<ac::core::ImreadModes>(core, "ImreadModes")
        .value("IMREAD_UNCHANGED", ac::core::IMREAD_UNCHANGED)
        .value("IMREAD_GRAYSCALE", ac::core::IMREAD_GRAYSCALE)
        .value("IMREAD_COLOR", ac::core::IMREAD_COLOR)
        .value("IMREAD_RGB", ac::core::IMREAD_RGB)
        .value("IMREAD_RGBA", ac::core::IMREAD_RGBA)
        .export_values();

    core.def("imread", [](const char* filename, ac::core::ImreadModes mode) {
        auto img = ac::core::imread(filename, mode);
        return py::array_t<std::uint8_t>{
            img.channels() == 1 ? py::array::ShapeContainer{ img.height(), img.width() } : py::array::ShapeContainer{ img.height(), img.width(), img.channels() },
            img.channels() == 1 ? py::array::StridesContainer{ img.stride(), img.pixelSize() } : py::array::StridesContainer{ img.stride(), img.pixelSize(), img.elementSize() },
            img.data(),
            py::capsule{ new ac::core::Image{ img }, [](void* v) { delete static_cast<ac::core::Image*>(v); } }
        };
    }, py::arg("filename"), py::arg("mode") = ac::core::IMREAD_UNCHANGED);

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

    specs.attr("ModelList") = makeTuple(ac::specs::ModelList);
    specs.attr("ModelDescriptionList") = makeTuple(ac::specs::ModelDescriptionList);
    specs.attr("ProcessorList") = makeTuple(ac::specs::ProcessorList);
    specs.attr("ProcessorDescriptionList") = makeTuple(ac::specs::ProcessorDescriptionList);
}
