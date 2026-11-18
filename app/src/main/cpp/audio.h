#pragma once

namespace tv {

enum class Sfx { Hover, Select, Grab, Release, Switch, Step };

// Procedural UI sounds through AAudio (no audio assets). Best effort: if the stream
// cannot be opened every call becomes a no-op.
class Audio {
public:
    bool init();
    void play(Sfx s);
    void shutdown();
    bool ok() const { return stream_ != nullptr; }

private:
    void* stream_ = nullptr;
    void* state_ = nullptr;
};

}  // namespace tv
