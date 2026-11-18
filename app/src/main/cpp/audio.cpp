#include "audio.h"

#include <aaudio/AAudio.h>
#include <android/log.h>

#include <atomic>
#include <cmath>

namespace tv {
namespace {

struct Voice { float f0, f1, ms, amp; };
// Hover, Select, Grab, Release, Switch, Step
const Voice kVoices[6] = {
    {1900.0f, 2100.0f, 35.0f, 0.10f},
    {880.0f, 1320.0f, 170.0f, 0.24f},
    {320.0f, 380.0f, 90.0f, 0.20f},
    {260.0f, 200.0f, 90.0f, 0.15f},
    {660.0f, 990.0f, 210.0f, 0.22f},
    {1046.0f, 1568.0f, 260.0f, 0.20f},
};

struct State {
    std::atomic<int> pending{-1};
    int sampleRate = 48000;
    // audio-thread only
    float phase = 0.0f;
    int total = 0, remaining = 0;
    Voice v{0, 0, 0, 0};
};

aaudio_data_callback_result_t onAudio(AAudioStream*, void* user, void* data, int32_t frames) {
    State* s = static_cast<State*>(user);
    float* out = static_cast<float*>(data);
    for (int32_t i = 0; i < frames; ++i) {
        if (s->remaining <= 0) {
            const int p = s->pending.exchange(-1);
            if (p >= 0 && p < 6) {
                s->v = kVoices[p];
                s->total = s->remaining = static_cast<int>(s->v.ms * 0.001f * static_cast<float>(s->sampleRate));
                s->phase = 0.0f;
            }
        }
        float sample = 0.0f;
        if (s->remaining > 0) {
            const float t = 1.0f - static_cast<float>(s->remaining) / static_cast<float>(s->total);  // 0..1
            const float f = s->v.f0 + (s->v.f1 - s->v.f0) * t;
            const float attack = std::min(1.0f, t * s->total / (0.003f * static_cast<float>(s->sampleRate)));
            const float env = attack * (1.0f - t) * (1.0f - t);
            s->phase += 6.2831853f * f / static_cast<float>(s->sampleRate);
            if (s->phase > 6.2831853f) s->phase -= 6.2831853f;
            sample = std::sin(s->phase) * env * s->v.amp;
            --s->remaining;
        }
        out[i] = sample;
    }
    return AAUDIO_CALLBACK_RESULT_CONTINUE;
}

}  // namespace

bool Audio::init() {
    AAudioStreamBuilder* b = nullptr;
    if (AAudio_createStreamBuilder(&b) != AAUDIO_OK || !b) return false;
    State* st = new State();
    AAudioStreamBuilder_setDirection(b, AAUDIO_DIRECTION_OUTPUT);
    AAudioStreamBuilder_setPerformanceMode(b, AAUDIO_PERFORMANCE_MODE_LOW_LATENCY);
    AAudioStreamBuilder_setSharingMode(b, AAUDIO_SHARING_MODE_SHARED);
    AAudioStreamBuilder_setFormat(b, AAUDIO_FORMAT_PCM_FLOAT);
    AAudioStreamBuilder_setChannelCount(b, 1);
    AAudioStreamBuilder_setSampleRate(b, 48000);
    AAudioStreamBuilder_setDataCallback(b, onAudio, st);
    AAudioStream* stream = nullptr;
    const aaudio_result_t r = AAudioStreamBuilder_openStream(b, &stream);
    AAudioStreamBuilder_delete(b);
    if (r != AAUDIO_OK || !stream) { delete st; return false; }
    st->sampleRate = AAudioStream_getSampleRate(stream);
    if (AAudioStream_requestStart(stream) != AAUDIO_OK) {
        AAudioStream_close(stream);
        delete st;
        return false;
    }
    stream_ = stream;
    state_ = st;
    return true;
}

void Audio::play(Sfx s) {
    if (state_) static_cast<State*>(state_)->pending.store(static_cast<int>(s));
}

void Audio::shutdown() {
    if (stream_) {
        AAudioStream_requestStop(static_cast<AAudioStream*>(stream_));
        AAudioStream_close(static_cast<AAudioStream*>(stream_));
    }
    delete static_cast<State*>(state_);
    stream_ = state_ = nullptr;
}

}  // namespace tv
