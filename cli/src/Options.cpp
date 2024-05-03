#include <cstdlib>

#include <CLI/CLI.hpp>

#include "Options.hpp"

Options parse(const int argc, const char* argv[]) noexcept
{
    Options options{};
    CLI::App app{"Anime4KCPP: A high performance anime upscaler."};

    app.fallthrough();

    app.add_option("-i,--input,input", options.input, "input file.")
        ->check(CLI::ExistingFile);
    app.add_option("-o,--output,output", options.output, "output file.");

    app.add_option("-m,--model", options.model, "acnet[-hdn][-0|1|2|3].")
        ->transform(CLI::detail::to_lower)
        ->capture_default_str();
    app.add_option("-p,--processor", options.processor, "processor for upscaling, use `-l` to list.")
        ->transform(CLI::detail::to_lower)
        ->capture_default_str();
    app.add_option("-d,--device", options.device, "device index for processor, use `-l` to list.")
        ->capture_default_str();
    app.add_option("-f,--factor", options.factor, "factor for upscaling.")
        ->capture_default_str();

    app.add_flag("-l,--list", options.list, "list processor info.");
    app.add_flag("-v,--version", options.version, "show version info.");

    auto video = app.add_subcommand("video", "video process module");
    video->add_option("--decoder", options.video.decoder, "decoder to use");
    video->add_option("--encoder", options.video.encoder, "decoder to use");
    video->add_option("--bitrate", options.video.bitrate, "bitrate for encoding, kbit/s");

    try { app.parse(argc, argv); } catch(const CLI::ParseError &e) { std::exit(app.exit(e)); }

    options.video.enable = video->parsed();

    return options;
}
