#ifndef _ENCODER_UTIL_H
#define _ENCODER_UTIL_H

#include <iostream>
#include <fstream>
#include <vector>
#include "particle_wrapper.h"
#include "camera.h"
#include <boost/python/numpy.hpp>

namespace np = boost::python::numpy;
namespace p = boost::python;

extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavutil/opt.h>
    #include <libavutil/imgutils.h>
}

struct codec_settings_t {
    bool use_bitrate;
    std::string codec_name, preset;
    int crf;
    int64_t bitrate;
    codec_settings_t() { }
    codec_settings_t(std::string codec_name, int64_t bitrate) :
        use_bitrate(true), codec_name(codec_name), preset(""), crf(-1), bitrate(bitrate) { }
    codec_settings_t(std::string codec_name, std::string preset, int crf) :
        use_bitrate(false), codec_name(codec_name), preset(preset), crf(crf), bitrate(-1) { }
};

class video_encoder {
private:
    std::string filename;
    std::ofstream filehandle;

    int width, height, framerate, pts=0;
    codec_settings_t codec_settings;

    const AVCodec *codec = nullptr;
    AVCodecContext *context = nullptr;
    AVPacket *packet = nullptr;
    AVFrame *frame = nullptr;

    void encode(AVFrame* frame_to_write);

public:
    video_encoder(const std::string& filename, int width, int height, int framerate, const codec_settings_t& codec_settings, int gop_size=10);

    ~video_encoder();

    video_encoder(const video_encoder& other) = delete;
    video_encoder(video_encoder&& other) = delete;
    video_encoder& operator=(const video_encoder& other) = delete;
    video_encoder& operator=(video_encoder&& other) = delete;

    np::ndarray generate_pixels(particle_wrapper& wrapper, const camera_settings_t& camera);
    void write_array(np::ndarray& arr);
    void write_from_wrapper(particle_wrapper& wrapper, const camera_settings_t& camera);
    void update_pixels(const std::vector<std::vector<uint8_t>>& data);
    void write_frame();
};

#ifdef av_err2str
#undef av_err2str
#include <string>
av_always_inline std::string av_err2string(int errnum) {
    char str[AV_ERROR_MAX_STRING_SIZE];
    return av_make_error_string(str, AV_ERROR_MAX_STRING_SIZE, errnum);
}
#define av_err2str(err) av_err2string(err).c_str()
#endif  // av_err2str

#endif