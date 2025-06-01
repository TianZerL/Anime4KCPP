#include <cstdlib>

#include <CLI/CLI.hpp>

#include "Options.hpp"

Options parse(const int argc, const char* const* argv) noexcept
{
    Options options{};
    CLI::App app{ "Anime4KCPP: A high performance anime upscaler." };

    app.fallthrough();

    app.add_option("-i,--input,input", options.inputs, "input files.");
    app.add_option("-o,--output", options.outputs, "output files.");

    app.add_option("-m,--model", options.model, "model name, use '--lm' to list")
        ->transform(CLI::detail::to_lower)
        ->capture_default_str();
    app.add_option("-p,--processor", options.processor, "processor for upscaling, use `--lp` to list.")
        ->transform(CLI::detail::to_lower)
        ->capture_default_str();
    app.add_option("-d,--device", options.device, "device index for processor, use `--ld` to list.")
        ->capture_default_str();
    app.add_option("-f,--factor", options.factor, "factor for upscaling.")
        ->capture_default_str();

    app.add_flag("--lm,--models", options.list.models, "list models.");
    app.add_flag("--lp,--processors", options.list.processors, "list processor info.");
    app.add_flag("-l,--ld,--devices", options.list.devices, "list device info.");
    app.add_flag("-v,--lv,--version", options.list.version, "show version info.");

    auto video = app.add_subcommand("video", "video processing");
    video->add_option("--vd,--decoder", options.video.decoder, "decoder to use");
    video->add_option("--vf,--format", options.video.format, "decode format");
    video->add_option("--ve,--encoder", options.video.encoder, "encoder to use");
    video->add_option("--vb,--bitrate", options.video.bitrate, "bitrate for encoding, kbit/s");

    try { app.parse(argc, argv); }
    catch (const CLI::ParseError& e) { std::exit(app.exit(e)); }

    options.video.enable = video->parsed();

    return options;
}
