#include "encoder_util.h"

video_encoder::video_encoder(const std::string& filename, int width, int height, int framerate, const codec_settings_t& codec_settings, int gop_size)
{
    this->filename = filename;
    this->width = width;
    this->height = height;
    this->framerate = framerate;
    this->codec_settings = codec_settings;
    
    codec = avcodec_find_encoder_by_name(codec_settings.codec_name.c_str());
    if(!codec){
        std::cerr << "Codec '" << codec_settings.codec_name << "' not found" << std::endl;
        exit(1);
    }

    context = avcodec_alloc_context3(codec);
    if(!context){
        std::cerr << "Unable to allocate video codec context" << std::endl;
        exit(1);
    }

    packet = av_packet_alloc();
    if(!packet){
        std::cerr << "Unable to allocate packet" << std::endl;
        exit(1);
    }

    context->width = width;
    context->height = height;
    context->time_base = (AVRational){1, framerate};
    context->framerate = (AVRational){framerate, 1};

    context->gop_size = gop_size;
    context->max_b_frames = 1;
    context->pix_fmt = AV_PIX_FMT_YUV420P;

    if(codec_settings.use_bitrate) {
        std::cout << "Setting bitrate" << std::endl;
        context->bit_rate = codec_settings.bitrate;
    }else{
        std::cout << "Setting crf" << std::endl;
        av_opt_set(context->priv_data, "preset", codec_settings.preset.c_str(), 0);
        av_opt_set_int(context, "crf", codec_settings.crf, AV_OPT_SEARCH_CHILDREN);
    }

    int ret;
    ret = avcodec_open2(context, codec, nullptr);
    if(ret < 0){
        std::cerr << "Could not open codec: " << av_err2str(ret) << std::endl;
        exit(1);
    }

    this->filehandle.open(filename, std::ios::out | std::ios::binary);
    if(!this->filehandle){
        std::cerr << "Could not open " << filename << std::endl;
        exit(1);
    }

    frame = av_frame_alloc();
    if(!frame){
        std::cerr << "Could not allocate video frame" << std::endl;
        exit(1);
    }

    frame->format = context->pix_fmt;
    frame->width = context->width;
    frame->height = context->height;

    ret = av_frame_get_buffer(frame, 0);
    if(ret < 0) {
        std::cerr << "Could not allocate video frame data" << std::endl;
        exit(1);
    }

    for (int y = 0; y < context->height/2; y++) {
        for (int x = 0; x < context->width/2; x++) {
            frame->data[1][y * frame->linesize[1] + x] = 128;
            frame->data[2][y * frame->linesize[2] + x] = 128;
        }
    }
}

video_encoder::~video_encoder()
{
    encode(nullptr);

    filehandle.close();
    if(context != nullptr) avcodec_free_context(&context);
    if(frame != nullptr) av_frame_free(&frame);
    if(packet != nullptr) av_packet_free(&packet);
}

np::ndarray video_encoder::generate_pixels(particle_wrapper& wrapper, const camera_settings_t& camera){
    np::ndarray res = np::zeros(p::make_tuple(this->height, this->width), np::dtype::get_builtin<uint8_t>());
    
    for(int i = 0; i < wrapper.get_count(); i++){
        auto pos = wrapper.get_particle_position(i);
        int x = (pos.x - camera.center.x) * camera.zoom + context->width / 2;
        int y = (pos.y - camera.center.y) * camera.zoom + context->height / 2;

        if(x < 0 || y < 0) continue;
        if(x >= this->width || y >= this->height) continue;
        res[y][x] = 255;
    }

    return res;
}

void video_encoder::write_array(np::ndarray& arr){
    if(arr.get_nd() != 2) {
        std::cerr << "Invalid array strides" << std::endl;
        exit(1);
    }

    auto raw_data = arr.get_data();
    auto strides = arr.get_strides();

    auto st1 = strides[0];
    auto st2 = strides[1];

    for(int y = 0; y < this->height; y++){
        for(int x = 0; x < this->width; x++){
            frame->data[0][y * frame->linesize[0] + x] = *(raw_data + y * st1 + x * st2);
        }
    }
    write_frame();
}

void video_encoder::write_from_wrapper(particle_wrapper& wrapper, const camera_settings_t& camera){
    for(int y = 0; y < this->height; y++){
        for(int x = 0; x < this->width; x++){
            frame->data[0][y * frame->linesize[0] + x] = 0;
        }
    }

    for(int i = 0; i < wrapper.get_count(); i++){
        auto pos = wrapper.get_particle_position(i);
        int x = (pos.x - camera.center.x) * camera.zoom + width / 2;
        int y = (pos.y - camera.center.y) * camera.zoom + height / 2;

        if(x < 0 || y < 0) continue;
        if(x >= width || y >= height) continue;
        frame->data[0][y * frame->linesize[0] + x] = 255;
    }
    write_frame();
}

void video_encoder::update_pixels(const std::vector<std::vector<uint8_t>>& data){
    for(int y = 0; y < context->height; y++){
        for(int x = 0; x < context->width; x++){
            frame->data[0][y * frame->linesize[0] + x] = data[x][y];
        }
    }
}

void video_encoder::encode(AVFrame* frame_to_write)
{
    int ret;

    frame->pts = pts;
    packet->pts = pts;
    pts++;

    ret = avcodec_send_frame(context, frame_to_write);
    if (ret < 0) {
        std::cerr << "Error sending a frame for encoding" << std::endl;
        exit(1);
    }

    while (ret >= 0) {
        ret = avcodec_receive_packet(context, packet);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return;
        else if (ret < 0) {
            std::cerr << "Error during encoding" << std::endl;
            exit(1);
        }

        filehandle.write(reinterpret_cast<char*>(packet->data), packet->size);
        av_packet_unref(packet);
    }
}

void video_encoder::write_frame()
{
    encode(this->frame);
}