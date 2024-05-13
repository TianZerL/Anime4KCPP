<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="zh_CN">
<context>
    <name>MainWindow</name>
    <message>
        <location filename="../ui/MainWindow.ui" line="14"/>
        <source>Anime4KCPP</source>
        <translation>Anime4KCPP</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="24"/>
        <location filename="../ui/MainWindow.ui" line="371"/>
        <source>Settings</source>
        <translation>设置</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="44"/>
        <source>decode hints</source>
        <translation>解码</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="58"/>
        <location filename="../ui/MainWindow.ui" line="68"/>
        <location filename="../ui/MainWindow.ui" line="161"/>
        <location filename="../src/MainWindow.cpp" line="180"/>
        <source>image</source>
        <translation>图像</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="78"/>
        <source>decoder</source>
        <translation>解码器</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="85"/>
        <source>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;Decoder name pass to ffmpeg&apos;s libav.&lt;/p&gt;&lt;p&gt;The available values depend on the version of the libav.&lt;/p&gt;&lt;p&gt;Typically including `&lt;span style=&quot; font-weight:700;&quot;&gt;h264&lt;/span&gt;`, `&lt;span style=&quot; font-weight:700;&quot;&gt;h264_cuvid&lt;/span&gt;` (Nvidia hwaccel), `&lt;span style=&quot; font-weight:700;&quot;&gt;h264_qsv&lt;/span&gt;` (Intel hwaccel), etc.&lt;/p&gt;&lt;p&gt;Normally, there is no need to specify a decoder, the most suitable one will be selected based on the input file.&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</source>
        <translation>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;解码器名称，传递给ffmpeg的libav库。&lt;/p&gt;&lt;p&gt;可取值取决于libav库的版本。&lt;/p&gt;&lt;p&gt;通常包括`&lt;span style=&quot; font-weight:700;&quot;&gt;h264&lt;/span&gt;`，`&lt;span style=&quot; font-weight:700;&quot;&gt;h264_cuvid&lt;/span&gt;` (Nvidia卡硬解)，`&lt;span style=&quot; font-weight:700;&quot;&gt;h264_qsv&lt;/span&gt;` (Intel显卡硬解)等。&lt;/p&gt;&lt;p&gt;通常无需指定，会根据文件后缀名自动选择最合适的解码器。&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="92"/>
        <source>encoder</source>
        <translation>编码器</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="99"/>
        <source>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;Encoder name pass to ffmpeg&apos;s libav.&lt;/p&gt;&lt;p&gt;The available values depend on the version of the libav.&lt;/p&gt;&lt;p&gt;Typically including `&lt;span style=&quot; font-weight:700;&quot;&gt;libopenh264&lt;/span&gt;`, `&lt;span style=&quot; font-weight:700;&quot;&gt;libx264&lt;/span&gt;`, `&lt;span style=&quot; font-weight:700;&quot;&gt;h264_qsv&lt;/span&gt;`, `&lt;span style=&quot; font-weight:700;&quot;&gt;h264_nvenc&lt;/span&gt;`, `&lt;span style=&quot; font-weight:700;&quot;&gt;h264_amf&lt;/span&gt;`, etc.&lt;/p&gt;&lt;p&gt;This will affect output codec, such as `&lt;span style=&quot; font-weight:700;&quot;&gt;libx265&lt;/span&gt;` (hevc), `&lt;span style=&quot; font-weight:700;&quot;&gt;av1_nvenc&lt;/span&gt;` (av1), `&lt;span style=&quot; font-weight:700;&quot;&gt;mpeg4&lt;/span&gt;` (mpeg4), etc. Check FFmpeg docs for more information.&lt;/p&gt;&lt;p&gt;This will also affect hardware acceleration, such as `&lt;span style=&quot; font-weight:700;&quot;&gt;hevc_nvenc&lt;/span&gt;` (Nvidia), `&lt;span style=&quot; font-weight:700;&quot;&gt;hevc_amf&lt;/span&gt;` (AMD), `&lt;span style=&quot; font-weight:700;&quot;&gt;hevc_qsv&lt;/span&gt;` (Intel), etc.&lt;/p&gt;&lt;p&gt;Leave blank to select the default encoder based on the file suffix.&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</source>
        <translation>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;编码器名称，传递给ffmpeg的libav库。&lt;/p&gt;&lt;p&gt;可取值取决于libav库的版本。&lt;/p&gt;&lt;p&gt;通常包括`&lt;span style=&quot; font-weight:700;&quot;&gt;libopenh264&lt;/span&gt;`，`&lt;span style=&quot; font-weight:700;&quot;&gt;libx264&lt;/span&gt;`，`&lt;span style=&quot; font-weight:700;&quot;&gt;h264_qsv&lt;/span&gt;`，`&lt;span style=&quot; font-weight:700;&quot;&gt;h264_nvenc&lt;/span&gt;`，`&lt;span style=&quot; font-weight:700;&quot;&gt;h264_amf&lt;/span&gt;`等。&lt;/p&gt;&lt;p&gt;取值会决定输出文件编码，比如`&lt;span style=&quot; font-weight:700;&quot;&gt;libx265&lt;/span&gt;` (hevc)，`&lt;span style=&quot; font-weight:700;&quot;&gt;av1_nvenc&lt;/span&gt;`  (av1)，`&lt;span style=&quot; font-weight:700;&quot;&gt;mpeg4&lt;/span&gt;` (mpeg4)等。可查看ffmpeg文档了解详细信息。&lt;/p&gt;&lt;p&gt;还会影响是否进行硬件加速编码，比如`&lt;span style=&quot; font-weight:700;&quot;&gt;hevc_nvenc&lt;/span&gt;` (Nvidia)，`&lt;span style=&quot; font-weight:700;&quot;&gt;hevc_amf&lt;/span&gt;` (AMD)，`&lt;span style=&quot; font-weight:700;&quot;&gt;hevc_qsv&lt;/span&gt;` (Intel)等。&lt;/p&gt;&lt;p&gt;留空将会根据文件后缀名选择默认编码器。&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="106"/>
        <location filename="../ui/MainWindow.ui" line="116"/>
        <location filename="../ui/MainWindow.ui" line="185"/>
        <location filename="../src/MainWindow.cpp" line="180"/>
        <source>video</source>
        <translation>视频</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="126"/>
        <source>format</source>
        <translation>格式</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="133"/>
        <source>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;Decode format pass to ffmpeg&apos;s libav.&lt;/p&gt;&lt;p&gt;Normally, there is no need to specify a format, but you may need to use `&lt;span style=&quot; font-weight:700;&quot;&gt;nv12&lt;/span&gt;` for qsv encoder.&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</source>
        <translation>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;解码格式，传递给ffmpeg的libav库。&lt;/p&gt;&lt;p&gt;通常无需指定，若使用qsv编码器可能需要指定为`&lt;span style=&quot; font-weight:700;&quot;&gt;nv12&lt;/span&gt;`。&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="140"/>
        <source>bitrate</source>
        <translation>比特率</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="147"/>
        <source>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;bitrate (&lt;span style=&quot; font-weight:700;&quot;&gt;kb/s&lt;/span&gt;) pass to ffmpeg&apos;s libav.&lt;/p&gt;&lt;p&gt;This will affect the size and quality of the output video.&lt;/p&gt;&lt;p&gt;Leave blank to automatically calculate based on the input video bitrate.&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</source>
        <translation>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;单位(&lt;span style=&quot; font-weight:700;&quot;&gt;kb/s&lt;/span&gt;)，传递给ffmpeg的libav库。&lt;/p&gt;&lt;p&gt;取值会影响编码视频的质量和大小。&lt;/p&gt;&lt;p&gt;留空将根据输入视频自动计算。&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="171"/>
        <location filename="../ui/MainWindow.ui" line="195"/>
        <source>select</source>
        <translation>选择</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="178"/>
        <location filename="../ui/MainWindow.ui" line="202"/>
        <location filename="../src/MainWindow.cpp" line="116"/>
        <source>open</source>
        <translation>打开</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="209"/>
        <source>encode hints</source>
        <translation>编码</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="216"/>
        <source>prefix</source>
        <translation>前缀</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="223"/>
        <source>suffix</source>
        <translation>后缀</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="230"/>
        <source>output path</source>
        <translation>输出路径</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="243"/>
        <source>Parameters</source>
        <translation>参数</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="249"/>
        <source>processor</source>
        <translation>处理器</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="259"/>
        <source>device</source>
        <translation>设备</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="266"/>
        <source>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;For CPU, it represents different acceleration methods.&lt;/p&gt;&lt;p&gt;For GPU, it represents the devices to use.&lt;/p&gt;&lt;p&gt;Check available value in menu: &lt;span style=&quot; font-weight:700;&quot;&gt;Help-&amp;gt;List devices&lt;/span&gt;.&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</source>
        <translation>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;对于CPU该值为加速方式。&lt;/p&gt;&lt;p&gt;对于GPU该值为使用的设备。&lt;/p&gt;&lt;p&gt;可在菜单中检查可用值：&lt;span style=&quot; font-weight:700;&quot;&gt;帮助&amp;gt;列出设备&lt;/span&gt;。&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="273"/>
        <source>factor</source>
        <translation>系数</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="280"/>
        <source>Upscaling factor for each dimension.</source>
        <translation>每个维度的放大系数。</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="287"/>
        <source>model</source>
        <translation>模型</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="294"/>
        <source>ACNet: Higher HDN levels mean better denoising, which may compromise details.</source>
        <translation>ACNet：HDN等级越高则降噪效果越好，但可能会导致细节丢失。</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="304"/>
        <source>add</source>
        <translation>添加</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="311"/>
        <source>clear</source>
        <translation>清除</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="318"/>
        <source>start</source>
        <translation>开始</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="325"/>
        <source>stop</source>
        <translation>停止</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="359"/>
        <source>File</source>
        <translation>文件</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="367"/>
        <source>Help</source>
        <translation>帮助</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="375"/>
        <source>Style</source>
        <translation>风格</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="392"/>
        <source>About</source>
        <translation>关于</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="397"/>
        <source>Add</source>
        <translation>添加</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="402"/>
        <source>Exit</source>
        <translation>退出</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="407"/>
        <source>List devices</source>
        <translation>列出设备</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="415"/>
        <source>Exit confirmation</source>
        <translation>退出确认</translation>
    </message>
    <message>
        <location filename="../src/MainWindow.cpp" line="109"/>
        <source>type</source>
        <translation>类型</translation>
    </message>
    <message>
        <location filename="../src/MainWindow.cpp" line="109"/>
        <source>status</source>
        <translation>状态</translation>
    </message>
    <message>
        <location filename="../src/MainWindow.cpp" line="109"/>
        <source>name</source>
        <translation>名称</translation>
    </message>
    <message>
        <location filename="../src/MainWindow.cpp" line="109"/>
        <source>output name</source>
        <translation>输出名称</translation>
    </message>
    <message>
        <location filename="../src/MainWindow.cpp" line="109"/>
        <source>path</source>
        <translation>路径</translation>
    </message>
    <message>
        <location filename="../src/MainWindow.cpp" line="117"/>
        <source>remove</source>
        <translation>移除</translation>
    </message>
    <message>
        <location filename="../src/MainWindow.cpp" line="181"/>
        <source>ready</source>
        <translation>就绪</translation>
    </message>
    <message>
        <location filename="../src/MainWindow.cpp" line="188"/>
        <source>input</source>
        <translation>输入</translation>
    </message>
    <message>
        <location filename="../src/MainWindow.cpp" line="188"/>
        <source>output</source>
        <translation>输出</translation>
    </message>
    <message>
        <location filename="../src/MainWindow.cpp" line="191"/>
        <source>successed</source>
        <translation>成功</translation>
    </message>
    <message>
        <location filename="../src/MainWindow.cpp" line="191"/>
        <source>failed</source>
        <translation>失败</translation>
    </message>
</context>
</TS>
