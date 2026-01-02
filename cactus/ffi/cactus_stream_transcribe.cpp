#include "cactus_ffi.h"
#include "cactus_utils.h"
#include <chrono>
#include <cstring>

using namespace cactus::engine;
using namespace cactus::ffi;

static std::string escape_json_string(const std::string& s) {
    std::string escaped;
    escaped.reserve(s.length());
    for (char c : s) {
        if (c == '"') escaped += "\\\"";
        else if (c == '\n') escaped += "\\n";
        else if (c == '\r') escaped += "\\r";
        else if (c == '\t') escaped += "\\t";
        else if (c == '\\') escaped += "\\\\";
        else escaped += c;
    }
    return escaped;
}

static std::string extract_json_string_value(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return "";

    size_t colon_pos = json.find(':', pos);
    if (colon_pos == std::string::npos) return "";

    size_t start_quote = json.find('"', colon_pos);
    if (start_quote == std::string::npos) return "";

    std::string result;
    size_t i = start_quote + 1;
    while (i < json.length()) {
        if (json[i] == '"' && (i == start_quote + 1 || json[i - 1] != '\\')) {
            break;
        }
        if (json[i] == '\\' && i + 1 < json.length()) {
            char next = json[i + 1];
            if (next == '"') {
                result += '"';
                i += 2;
            } else if (next == '\\') {
                result += '\\';
                i += 2;
            } else if (next == 'n') {
                result += '\n';
                i += 2;
            } else if (next == 'r') {
                result += '\r';
                i += 2;
            } else if (next == 't') {
                result += '\t';
                i += 2;
            } else {
                result += json[i];
                i++;
            }
        } else {
            result += json[i];
            i++;
        }
    }
    return result;
}

static size_t levenshtein_distance(const std::string& a, const std::string& b, size_t n) {
    std::vector<std::vector<size_t>> dp(n + 1, std::vector<size_t>(n + 1));

    for (size_t i = 0; i <= n; ++i) {
        dp[i][0] = i;
        dp[0][i] = i;
    }

    for (size_t i = 1; i <= n; ++i) {
        for (size_t j = 1; j <= n; ++j) {
            if (a[i - 1] == b[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = std::min({
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + 1
                });
            }
        }
    }

    return dp[n][n];
}

static bool fuzzy_match(const std::string& a, const std::string& b, size_t n, double threshold) {
    if (!n) return false;
    if (a.size() < n || b.size() < n) return false;

    size_t dist = levenshtein_distance(a, b, n);
    return 1.0 - static_cast<double>(dist) / static_cast<double>(n) >= threshold;
}

static std::string get_last_n_words(const std::string& text, size_t n) {
    if (text.empty() || n == 0) return "";

    size_t word_count = 0;
    bool in_word = false;

    for (size_t i = text.length(); i-- > 0; ) {
        bool is_space = std::isspace(text[i]);
        if (is_space && in_word) {
            ++word_count;
            in_word = false;
            if (word_count == n) {
                return text.substr(i + 1);
            }
        } else if (!is_space) {
            in_word = true;
        }
    }

    return text;
}

static void parse_stream_transcribe_process_options(const std::string& json, double& confirmation_threshold) {
    confirmation_threshold = 0.95;

    if (json.empty()) {
        return;
    }

    size_t pos = json.find("\"confirmation_threshold\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        confirmation_threshold = std::stod(json.substr(pos));
    }
}

struct CactusStreamTranscribeHandle {
    CactusModelHandle* model_handle;

    std::string confirmed;
    std::string partial;

    std::vector<uint8_t> audio_buffer;

    std::string previous_transcription;
    size_t previous_audio_buffer_size;
};

extern "C" {

cactus_stream_transcribe_t cactus_stream_transcribe_init(cactus_model_t model) {
    if (!model) {
        last_error_message = "Model not initialized. Check model path and files.";
        CACTUS_LOG_ERROR("stream_transcribe_init", last_error_message);
        return nullptr;
    }

    try {
        auto* model_handle = static_cast<CactusModelHandle*>(model);
        if (!model_handle->model) {
            last_error_message = "Invalid model handle.";
            CACTUS_LOG_ERROR("stream_transcribe_init", last_error_message);
            return nullptr;
        }

        auto* stream_handle = new CactusStreamTranscribeHandle();
        stream_handle->model_handle = model_handle;
        stream_handle->previous_audio_buffer_size = 0;

        CACTUS_LOG_INFO("stream_transcribe_init",
            "Stream transcription initialized for model: " << model_handle->model_name);

        return stream_handle;
    } catch (const std::exception& e) {
        last_error_message = "Exception during stream_transcribe_init: " + std::string(e.what());
        CACTUS_LOG_ERROR("stream_transcribe_init", last_error_message);
        return nullptr;
    } catch (...) {
        last_error_message = "Unknown exception during stream transcription initialization";
        CACTUS_LOG_ERROR("stream_transcribe_init", last_error_message);
        return nullptr;
    }
}

int cactus_stream_transcribe_insert(
    cactus_stream_transcribe_t stream,
    const uint8_t* pcm_buffer,
    size_t pcm_buffer_size
) {
    if (!stream) {
        last_error_message = "Stream not initialized.";
        CACTUS_LOG_ERROR("stream_transcribe_insert", last_error_message);
        return -1;
    }

    if (!pcm_buffer || pcm_buffer_size == 0) {
        last_error_message = "Invalid parameters: pcm_buffer or pcm_buffer_size";
        CACTUS_LOG_ERROR("stream_transcribe_insert", last_error_message);
        return -1;
    }

    try {
        auto* handle = static_cast<CactusStreamTranscribeHandle*>(stream);
        handle->audio_buffer.insert(handle->audio_buffer.end(), pcm_buffer,
                                    pcm_buffer + pcm_buffer_size);

        CACTUS_LOG_DEBUG("stream_transcribe_insert",
            "Inserted " << pcm_buffer_size << " bytes, buffer size: " << handle->audio_buffer.size());

        return 0;
    } catch (const std::exception& e) {
        last_error_message = "Exception during stream_transcribe_insert: " + std::string(e.what());
        CACTUS_LOG_ERROR("stream_transcribe_insert", last_error_message);
        return -1;
    } catch (...) {
        last_error_message = "Unknown exception during audio buffer insertion";
        CACTUS_LOG_ERROR("stream_transcribe_insert", last_error_message);
        return -1;
    }
}

int cactus_stream_transcribe_process(
    cactus_stream_transcribe_t stream,
    char* response_buffer,
    size_t buffer_size,
    const char* options_json
) {
    if (!stream) {
        last_error_message = "Stream not initialized.";
        CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
        return -1;
    }

    if (!response_buffer || buffer_size == 0) {
        last_error_message = "Invalid parameters: response_buffer or buffer_size";
        CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
        return -1;
    }

    try {
        auto* handle = static_cast<CactusStreamTranscribeHandle*>(stream);

        double confirmation_threshold;
        parse_stream_transcribe_process_options(
            options_json ? options_json : "",
            confirmation_threshold
        );

        char response_buffer_[4096];

        std::string prompt = "<|startofprev|>"
            + get_last_n_words(handle->confirmed, 200)
            + "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>";

        const int result = cactus_transcribe(
            handle->model_handle,
            nullptr,
            prompt.c_str(),
            response_buffer_,
            sizeof(response_buffer_),
            nullptr,
            nullptr,
            nullptr,
            handle->audio_buffer.data(),
            handle->audio_buffer.size());

        cactus_reset(handle->model_handle);

        if (result < 0) {
            last_error_message = "Transcription failed in stream process.";
            CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
            handle_error_response(last_error_message, response_buffer, buffer_size);
            return -1;
        }

        std::string json_str(response_buffer_);
        std::string response = extract_json_string_value(json_str, "response");
        std::string json_response = "{\"success\":true,\"confirmed\":\"" +
                                    escape_json_string(handle->confirmed) + "\",\"partial\":\"" +
                                    escape_json_string(response) + "\"}";

        if (json_response.length() >= buffer_size) {
            last_error_message = "Response buffer too small";
            CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
            handle_error_response(last_error_message, response_buffer, buffer_size);
            return -1;
        }

        std::strcpy(response_buffer, json_response.c_str());

        const size_t n = std::min(handle->previous_transcription.size(), response.size());
        if (fuzzy_match(handle->previous_transcription, response, n, confirmation_threshold)) {
            handle->audio_buffer = std::vector<uint8_t>(
                handle->audio_buffer.begin() + handle->previous_audio_buffer_size,
                handle->audio_buffer.end()
            );
            handle->confirmed += handle->previous_transcription;
        }

        handle->previous_transcription = response;
        handle->previous_audio_buffer_size = handle->audio_buffer.size();

        return static_cast<int>(json_response.length());
    } catch (const std::exception& e) {
        last_error_message = "Exception during stream_transcribe_process: " + std::string(e.what());
        CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
        handle_error_response(e.what(), response_buffer, buffer_size);
        return -1;
    } catch (...) {
        last_error_message = "Unknown exception during stream transcription processing";
        CACTUS_LOG_ERROR("stream_transcribe_process", last_error_message);
        handle_error_response("Unknown error during stream processing", response_buffer, buffer_size);
        return -1;
    }
}

void cactus_stream_transcribe_destroy(cactus_stream_transcribe_t stream) {
    if (stream) delete static_cast<CactusStreamTranscribeHandle*>(stream);
}

}
