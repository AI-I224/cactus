#pragma once

#ifdef HAVE_SDL2

#include <SDL.h>
#include <SDL_audio.h>
#include <vector>
#include <mutex>
#include <atomic>
#include <cstring>
#include <iostream>

class AudioCapture {
public:
    AudioCapture(int len_ms = 10000)
        : m_len_ms(len_ms)
        , m_running(false)
        , m_dev_id_in(0)
        , m_audio_pos(0)
        , m_audio_len(0)
        , m_total_samples_received(0) {
    }

    ~AudioCapture() {
        if (m_dev_id_in) {
            SDL_CloseAudioDevice(m_dev_id_in);
        }
    }

    bool init(int capture_id, int sample_rate) {
        if (SDL_Init(SDL_INIT_AUDIO) < 0) {
            std::cerr << "SDL_Init failed: " << SDL_GetError() << std::endl;
            return false;
        }

        SDL_SetHintWithPriority(SDL_HINT_AUDIO_RESAMPLING_MODE, "medium", SDL_HINT_OVERRIDE);

        m_audio.resize((m_len_ms * sample_rate) / 1000);

        int num_devices = SDL_GetNumAudioDevices(SDL_TRUE);
        std::cout << "\nAvailable audio capture devices:\n";
        for (int i = 0; i < num_devices; i++) {
            std::cout << "  [" << i << "] " << SDL_GetAudioDeviceName(i, SDL_TRUE) << "\n";
        }

        if (capture_id >= num_devices) {
            std::cerr << "Invalid capture device ID: " << capture_id << std::endl;
            return false;
        }

        std::cout << "Selected device: [" << capture_id << "] "
                  << SDL_GetAudioDeviceName(capture_id, SDL_TRUE) << "\n\n";

        SDL_AudioSpec capture_spec_requested;
        SDL_zero(capture_spec_requested);

        capture_spec_requested.freq = sample_rate;
        capture_spec_requested.format = AUDIO_F32;
        capture_spec_requested.channels = 1;
        capture_spec_requested.samples = 1024;
        capture_spec_requested.callback = [](void* userdata, uint8_t* stream, int len) {
            AudioCapture* audio = static_cast<AudioCapture*>(userdata);
            audio->callback(stream, len);
        };
        capture_spec_requested.userdata = this;

        SDL_AudioSpec capture_spec_obtained;
        m_dev_id_in = SDL_OpenAudioDevice(
            SDL_GetAudioDeviceName(capture_id, SDL_TRUE),
            SDL_TRUE,
            &capture_spec_requested,
            &capture_spec_obtained,
            0
        );

        if (!m_dev_id_in) {
            std::cerr << "SDL_OpenAudioDevice failed: " << SDL_GetError() << std::endl;
            return false;
        }

        std::cout << "Audio capture initialized:\n"
                  << "  Sample rate: " << capture_spec_obtained.freq << " Hz\n"
                  << "  Channels: " << (int)capture_spec_obtained.channels << "\n"
                  << "  Samples: " << capture_spec_obtained.samples << "\n"
                  << "  Buffer length: " << m_len_ms << " ms\n";

        return true;
    }

    void resume() {
        if (!m_running && m_dev_id_in) {
            SDL_PauseAudioDevice(m_dev_id_in, 0);
            m_running = true;
        }
    }

    void pause() {
        if (m_running && m_dev_id_in) {
            SDL_PauseAudioDevice(m_dev_id_in, 1);
            m_running = false;
        }
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_audio_pos = 0;
        m_audio_len = 0;
    }

    size_t get(int duration_ms, std::vector<float>& result) {
        std::lock_guard<std::mutex> lock(m_mutex);

        const size_t n_samples = (duration_ms * m_audio.size()) / m_len_ms;
        if (n_samples > m_audio_len) {
            return 0;
        }

        result.resize(n_samples);

        size_t start_pos = (m_audio_pos + m_audio.size() - m_audio_len) % m_audio.size();
        for (size_t i = 0; i < n_samples; i++) {
            result[i] = m_audio[(start_pos + i) % m_audio.size()];
        }

        m_audio_len = (m_audio_len > n_samples) ? (m_audio_len - n_samples) : 0;

        return n_samples;
    }

    size_t get_all(std::vector<float>& result) {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (m_audio_len == 0) return 0;

        result.resize(m_audio_len);

        size_t start_pos = (m_audio_pos + m_audio.size() - m_audio_len) % m_audio.size();
        for (size_t i = 0; i < m_audio_len; i++) {
            result[i] = m_audio[(start_pos + i) % m_audio.size()];
        }

        size_t n_samples = m_audio_len;
        m_audio_len = 0;

        return n_samples;
    }

    bool is_running() const { return m_running; }

    size_t get_total_samples_received() const { return m_total_samples_received; }

    size_t get_buffer_length() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_audio_len;
    }

private:
    void callback(uint8_t* stream, int len) {
        const size_t n_samples = len / sizeof(float);
        const float* samples = reinterpret_cast<const float*>(stream);

        if (!m_running) return;

        std::lock_guard<std::mutex> lock(m_mutex);

        for (size_t i = 0; i < n_samples; i++) {
            m_audio[m_audio_pos] = samples[i];
            m_audio_pos = (m_audio_pos + 1) % m_audio.size();

            if (m_audio_len < m_audio.size()) {
                m_audio_len++;
            }
        }

        m_total_samples_received += n_samples;
    }

    int m_len_ms;
    std::atomic<bool> m_running;
    SDL_AudioDeviceID m_dev_id_in;

    std::vector<float> m_audio;
    size_t m_audio_pos;
    size_t m_audio_len;
    std::atomic<size_t> m_total_samples_received;
    mutable std::mutex m_mutex;
};

#endif
