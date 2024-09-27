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
        <location filename="../ui/MainWindow.ui" line="21"/>
        <location filename="../ui/MainWindow.ui" line="368"/>
        <source>Settings</source>
        <translation>设置</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="41"/>
        <source>decode hints</source>
        <translation>解码</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="55"/>
        <location filename="../ui/MainWindow.ui" line="65"/>
        <location filename="../ui/MainWindow.ui" line="158"/>
        <location filename="../src/MainWindow.cpp" line="185"/>
        <source>image</source>
        <translation>图像</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="75"/>
        <source>decoder</source>
        <translation>解码器</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="82"/>
        <source>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;Decoder name pass to ffmpeg&apos;s libav.&lt;/p&gt;&lt;p&gt;The available values depend on the version of the libav.&lt;/p&gt;&lt;p&gt;Typically including `&lt;span style=&quot; font-weight:700;&quot;&gt;h264&lt;/span&gt;`, `&lt;span style=&quot; font-weight:700;&quot;&gt;h264_cuvid&lt;/span&gt;` (Nvidia hwaccel), `&lt;span style=&quot; font-weight:700;&quot;&gt;h264_qsv&lt;/span&gt;` (Intel hwaccel), etc.&lt;/p&gt;&lt;p&gt;Normally, there is no need to specify a decoder, the most suitable one will be selected based on the input file.&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</source>
        <translation>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;解码器名称，传递给ffmpeg的libav库。&lt;/p&gt;&lt;p&gt;可取值取决于libav库的版本。&lt;/p&gt;&lt;p&gt;通常包括`&lt;span style=&quot; font-weight:700;&quot;&gt;h264&lt;/span&gt;`，`&lt;span style=&quot; font-weight:700;&quot;&gt;h264_cuvid&lt;/span&gt;` (Nvidia显卡硬解)，`&lt;span style=&quot; font-weight:700;&quot;&gt;h264_qsv&lt;/span&gt;` (Intel显卡硬解)等。&lt;/p&gt;&lt;p&gt;通常无需指定，会根据文件类型自动选择最合适的解码器。&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="89"/>
        <source>encoder</source>
        <translation>编码器</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="96"/>
        <source>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;Encoder name pass to ffmpeg&apos;s libav.&lt;/p&gt;&lt;p&gt;The available values depend on the version of the libav.&lt;/p&gt;&lt;p&gt;Typically including `&lt;span style=&quot; font-weight:700;&quot;&gt;libopenh264&lt;/span&gt;`, `&lt;span style=&quot; font-weight:700;&quot;&gt;libx264&lt;/span&gt;`, `&lt;span style=&quot; font-weight:700;&quot;&gt;h264_qsv&lt;/span&gt;`, `&lt;span style=&quot; font-weight:700;&quot;&gt;h264_nvenc&lt;/span&gt;`, `&lt;span style=&quot; font-weight:700;&quot;&gt;h264_amf&lt;/span&gt;`, etc.&lt;/p&gt;&lt;p&gt;This will affect output codec, such as `&lt;span style=&quot; font-weight:700;&quot;&gt;libx265&lt;/span&gt;` (hevc), `&lt;span style=&quot; font-weight:700;&quot;&gt;av1_nvenc&lt;/span&gt;` (av1), `&lt;span style=&quot; font-weight:700;&quot;&gt;mpeg4&lt;/span&gt;` (mpeg4), etc. Check FFmpeg docs for more information.&lt;/p&gt;&lt;p&gt;This will also affect hardware acceleration, such as `&lt;span style=&quot; font-weight:700;&quot;&gt;hevc_nvenc&lt;/span&gt;` (Nvidia), `&lt;span style=&quot; font-weight:700;&quot;&gt;hevc_amf&lt;/span&gt;` (AMD), `&lt;span style=&quot; font-weight:700;&quot;&gt;hevc_qsv&lt;/span&gt;` (Intel), etc.&lt;/p&gt;&lt;p&gt;Leave blank to select the default encoder based on the file suffix.&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</source>
        <translation>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;编码器名称，传递给ffmpeg的libav库。&lt;/p&gt;&lt;p&gt;可取值取决于libav库的版本。&lt;/p&gt;&lt;p&gt;通常包括`&lt;span style=&quot; font-weight:700;&quot;&gt;libopenh264&lt;/span&gt;`，`&lt;span style=&quot; font-weight:700;&quot;&gt;libx264&lt;/span&gt;`，`&lt;span style=&quot; font-weight:700;&quot;&gt;h264_qsv&lt;/span&gt;`，`&lt;span style=&quot; font-weight:700;&quot;&gt;h264_nvenc&lt;/span&gt;`，`&lt;span style=&quot; font-weight:700;&quot;&gt;h264_amf&lt;/span&gt;`等。&lt;/p&gt;&lt;p&gt;取值会决定输出文件编码，比如`&lt;span style=&quot; font-weight:700;&quot;&gt;libx265&lt;/span&gt;` (hevc)，`&lt;span style=&quot; font-weight:700;&quot;&gt;av1_nvenc&lt;/span&gt;`  (av1)，`&lt;span style=&quot; font-weight:700;&quot;&gt;mpeg4&lt;/span&gt;` (mpeg4)等。可查看ffmpeg文档了解详细信息。&lt;/p&gt;&lt;p&gt;还会影响是否进行硬件加速编码，比如`&lt;span style=&quot; font-weight:700;&quot;&gt;hevc_nvenc&lt;/span&gt;` (Nvidia)，`&lt;span style=&quot; font-weight:700;&quot;&gt;hevc_amf&lt;/span&gt;` (AMD)，`&lt;span style=&quot; font-weight:700;&quot;&gt;hevc_qsv&lt;/span&gt;` (Intel)等。&lt;/p&gt;&lt;p&gt;留空将会根据文件后缀名选择默认编码器。&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="103"/>
        <location filename="../ui/MainWindow.ui" line="113"/>
        <location filename="../ui/MainWindow.ui" line="182"/>
        <location filename="../src/MainWindow.cpp" line="185"/>
        <source>video</source>
        <translation>视频</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="123"/>
        <source>format</source>
        <translation>格式</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="130"/>
        <source>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;Decode format pass to ffmpeg&apos;s libav.&lt;/p&gt;&lt;p&gt;Normally, there is no need to specify a format, but you may need to use `&lt;span style=&quot; font-weight:700;&quot;&gt;nv12&lt;/span&gt;` for qsv encoder.&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</source>
        <translation>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;解码格式，传递给ffmpeg的libav库。&lt;/p&gt;&lt;p&gt;通常无需指定，若使用qsv编码器可能需要指定为`&lt;span style=&quot; font-weight:700;&quot;&gt;nv12&lt;/span&gt;`。&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="137"/>
        <source>bitrate</source>
        <translation>比特率</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="144"/>
        <source>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;bitrate (&lt;span style=&quot; font-weight:700;&quot;&gt;kb/s&lt;/span&gt;) pass to ffmpeg&apos;s libav.&lt;/p&gt;&lt;p&gt;This will affect the size and quality of the output video.&lt;/p&gt;&lt;p&gt;Leave blank to automatically calculate based on the input video bitrate.&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</source>
        <translation>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;单位(&lt;span style=&quot; font-weight:700;&quot;&gt;kb/s&lt;/span&gt;)，传递给ffmpeg的libav库。&lt;/p&gt;&lt;p&gt;取值会影响编码视频的质量和大小。&lt;/p&gt;&lt;p&gt;留空将根据输入视频自动计算。&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="168"/>
        <location filename="../ui/MainWindow.ui" line="192"/>
        <source>select</source>
        <translation>选择</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="175"/>
        <location filename="../ui/MainWindow.ui" line="199"/>
        <location filename="../src/MainWindow.cpp" line="122"/>
        <source>open</source>
        <translation>打开</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="206"/>
        <source>encode hints</source>
        <translation>编码</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="213"/>
        <source>prefix</source>
        <translation>前缀</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="220"/>
        <source>suffix</source>
        <translation>后缀</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="227"/>
        <source>output path</source>
        <translation>输出路径</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="240"/>
        <source>Parameters</source>
        <translation>参数</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="246"/>
        <source>processor</source>
        <translation>处理器</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="256"/>
        <source>device</source>
        <translation>设备</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="263"/>
        <source>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;For CPU, it represents different acceleration methods.&lt;/p&gt;&lt;p&gt;For GPU, it represents the devices to use.&lt;/p&gt;&lt;p&gt;Check available value in menu: &lt;span style=&quot; font-weight:700;&quot;&gt;Help-&amp;gt;List devices&lt;/span&gt;.&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</source>
        <translation>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;对于CPU该值为加速方式。&lt;/p&gt;&lt;p&gt;对于GPU该值为使用的设备。&lt;/p&gt;&lt;p&gt;可在菜单中检查可用值：&lt;span style=&quot; font-weight:700;&quot;&gt;帮助&amp;gt;列出设备&lt;/span&gt;。&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="270"/>
        <source>factor</source>
        <translation>系数</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="277"/>
        <source>Upscaling factor for each dimension.</source>
        <translation>每个维度的放大系数。</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="284"/>
        <source>model</source>
        <translation>模型</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="291"/>
        <source>ACNet: Higher HDN levels mean better denoising, which may compromise details.</source>
        <translation>ACNet：HDN等级越高则降噪效果越好，但可能会导致细节丢失。</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="301"/>
        <source>add</source>
        <translation>添加</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="308"/>
        <source>clear</source>
        <translation>清除</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="315"/>
        <source>start</source>
        <translation>开始</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="322"/>
        <source>stop</source>
        <translation>停止</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="356"/>
        <source>File</source>
        <translation>文件</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="364"/>
        <source>Help</source>
        <translation>帮助</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="372"/>
        <source>Style</source>
        <translation>风格</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="390"/>
        <location filename="../src/MainWindow.cpp" line="304"/>
        <source>About</source>
        <translation>关于</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="395"/>
        <source>Add</source>
        <translation>添加</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="400"/>
        <location filename="../src/MainWindow.cpp" line="223"/>
        <source>Exit</source>
        <translation>退出</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="405"/>
        <source>List devices</source>
        <translation>列出设备</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="413"/>
        <location filename="../src/MainWindow.cpp" line="223"/>
        <source>Exit confirmation</source>
        <translation>退出确认</translation>
    </message>
    <message>
        <location filename="../ui/MainWindow.ui" line="418"/>
        <location filename="../src/MainWindow.cpp" line="278"/>
        <source>License</source>
        <translation>许可证</translation>
    </message>
    <message>
        <location filename="../src/MainWindow.cpp" line="115"/>
        <source>type</source>
        <translation>类型</translation>
    </message>
    <message>
        <location filename="../src/MainWindow.cpp" line="115"/>
        <source>status</source>
        <translation>状态</translation>
    </message>
    <message>
        <location filename="../src/MainWindow.cpp" line="115"/>
        <source>name</source>
        <translation>名称</translation>
    </message>
    <message>
        <location filename="../src/MainWindow.cpp" line="115"/>
        <source>output name</source>
        <translation>输出名称</translation>
    </message>
    <message>
        <location filename="../src/MainWindow.cpp" line="115"/>
        <source>path</source>
        <translation>路径</translation>
    </message>
    <message>
        <location filename="../src/MainWindow.cpp" line="123"/>
        <source>remove</source>
        <translation>移除</translation>
    </message>
    <message>
        <location filename="../src/MainWindow.cpp" line="186"/>
        <source>ready</source>
        <translation>就绪</translation>
    </message>
    <message>
        <location filename="../src/MainWindow.cpp" line="191"/>
        <source>input</source>
        <translation>输入</translation>
    </message>
    <message>
        <location filename="../src/MainWindow.cpp" line="191"/>
        <source>output</source>
        <translation>输出</translation>
    </message>
    <message>
        <location filename="../src/MainWindow.cpp" line="195"/>
        <source>successed</source>
        <translation>成功</translation>
    </message>
    <message>
        <location filename="../src/MainWindow.cpp" line="195"/>
        <source>failed</source>
        <translation>失败</translation>
    </message>
    <message>
        <location filename="../src/MainWindow.cpp" line="266"/>
        <source>Devices</source>
        <translation>设备</translation>
    </message>
    <message>
        <location filename="../src/MainWindow.cpp" line="322"/>
        <source>Anime4KCPP: A high performance anime upscaler</source>
        <translation>Anime4KCPP：高性能动漫超分工具</translation>
    </message>
    <message>
        <location filename="../src/MainWindow.cpp" line="323"/>
        <source>core version</source>
        <translation>核心版本</translation>
    </message>
    <message>
        <location filename="../src/MainWindow.cpp" line="324"/>
        <source>video module</source>
        <translation>视频模块</translation>
    </message>
    <message>
        <location filename="../src/MainWindow.cpp" line="325"/>
        <source>build date</source>
        <translation>构建日期</translation>
    </message>
    <message>
        <location filename="../src/MainWindow.cpp" line="326"/>
        <source>toolchain</source>
        <translation>工具链</translation>
    </message>
    <message>
        <location filename="../src/MainWindow.cpp" line="327"/>
        <source>Copyright</source>
        <translation>版权所有</translation>
    </message>
    <message>
        <location filename="../src/MainWindow.cpp" line="328"/>
        <source>disabled</source>
        <translation>不可用</translation>
    </message>
</context>
</TS>
