#include <cstdlib>

#include <CLI/CLI.hpp>

#include "Options.hpp"

Options parse(const int argc, const char* const* argv) noexcept
{
    Options options{};
    CLI::App app{ "Anime4KCPP: A high performance anime upscaler.", "ac_cli"};

    app.fallthrough();

    app.add_option("-i,--input,input", options.inputs, "Input files.");
    app.add_option("-o,--output", options.outputs, "Output files.");

    app.add_option("-m,--model", options.model, "Model name, use '--lm' to list")
        ->transform(CLI::detail::to_lower)
        ->capture_default_str();
    app.add_option("-p,--processor", options.processor, "Processor for upscaling, use `--lp` to list.")
        ->transform(CLI::detail::to_lower)
        ->capture_default_str();
    app.add_option("-d,--device", options.device, "Device index for processor, use `--ld` to list.")
        ->capture_default_str();
    app.add_option("-f,--factor", options.factor, "Factor for upscaling.")
        ->capture_default_str();

    app.add_flag("--lm,--models", options.list.models, "List models.");
    app.add_flag("--lp,--processors", options.list.processors, "List processor info.");
    app.add_flag("-l,--ld,--devices", options.list.devices, "List device info.");
    app.add_flag("-v,--lv,--version", options.list.version, "Show version info.");

    auto video = app.add_subcommand("video", "Video processing");
    video->add_option("--vd,--decoder", options.video.decoder, "Decoder to use");
    video->add_option("--vf,--format", options.video.format, "Decode format");
    video->add_option("--ve,--encoder", options.video.encoder, "Encoder to use");
    video->add_option("--vb,--bitrate", options.video.bitrate, "Bitrate for encoding, kbit/s");

    app.footer("Use 'ac_cli video -h' to get help for video processing.");

    try { app.parse(argc, argv); if (!options.list && options.inputs.empty()) throw CLI::CallForHelp(); }
    catch (const CLI::ParseError& e) { std::exit(app.exit(e)); }

    options.video.enable = video->parsed();

    return options;
}
