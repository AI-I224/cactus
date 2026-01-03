# Cactus Engine FFI Documentation

The Cactus Engine provides a clean C FFI (Foreign Function Interface) for integrating the LLM inference engine into various applications. This documentation covers all available functions, their parameters, and usage examples.

## Getting Started

Before using the Cactus Engine, you need to download model weights:

```bash
# Setup the environment
./setup

# Download model weights (converts HuggingFace models to Cactus format)
cactus download LiquidAI/LFM2-1.2B

# Or download a vision-language model
cactus download LiquidAI/LFM2-VL-450M

# Or download a whisper model for transcription
cactus download openai/whisper-small
```

Weights are saved to the `weights/` directory and can be loaded using `cactus_init()`.

## Types

### `cactus_model_t`
An opaque pointer type representing a loaded model instance. This handle is used throughout the API to reference a specific model.

```c
typedef void* cactus_model_t;
```

### `cactus_token_callback`
Callback function type for streaming token generation. Called for each generated token during completion.

```c
typedef void (*cactus_token_callback)(
    const char* token,      // The generated token text
    uint32_t token_id,      // The token's ID in the vocabulary
    void* user_data         // User-provided context data
);
```

## Core Functions

### `cactus_init`
Initializes a model from disk and prepares it for inference.

```c
cactus_model_t cactus_init(
    const char* model_path,   // Path to the model directory
    size_t context_size,      // Maximum context size (e.g., 2048)
    const char* corpus_dir    // Optional path to corpus directory for RAG (can be NULL)
);
```

**Returns:** Model handle on success, NULL on failure

**Example:**
```c
// Basic initialization
cactus_model_t model = cactus_init("../../weights/qwen3-600m", 2048, NULL);
if (!model) {
    fprintf(stderr, "Failed to initialize model\n");
    return -1;
}

// With RAG corpus (only works om LFM2 RAG version for now)
cactus_model_t rag_model = cactus_init("../../weights/lfm2-rag", 512, "./documents");
```

### `cactus_complete`
Performs text completion with optional streaming and tool support.

```c
int cactus_complete(
    cactus_model_t model,           // Model handle
    const char* messages_json,      // JSON array of messages
    char* response_buffer,          // Buffer for response JSON
    size_t buffer_size,             // Size of response buffer
    const char* options_json,       // Optional generation options (can be NULL)
    const char* tools_json,         // Optional tools definition (can be NULL)
    cactus_token_callback callback, // Optional streaming callback (can be NULL)
    void* user_data                 // User data for callback (can be NULL)
);
```

**Returns:** Number of bytes written to response_buffer on success, negative value on error

**Message Format:**
```json
[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is your name?"}
]
```

**Messages with Images (for VLM models):**
```json
[
    {"role": "user", "content": "Describe this image", "images": ["/path/to/image.jpg"]}
]
```

**Options Format:**
```json
{
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "stop_sequences": ["<|im_end|>", "<end_of_turn>"]
}
```

**Response Format:**
```json
{
    "success": true,
    "response": "I am an AI assistant.",
    "time_to_first_token_ms": 150.5,
    "total_time_ms": 1250.3,
    "tokens_per_second": 45.2,
    "prompt_tokens": 25,
    "completion_tokens": 8
}
```

**Response with Function Call:**
```json
{
    "success": true,
    "response": "",
    "function_calls": [
        {
            "name": "get_weather",
            "arguments": "{\"location\": \"San Francisco, CA, USA\"}"
        }
    ],
    "time_to_first_token_ms": 120.0,
    "total_time_ms": 450.5,
    "tokens_per_second": 38.5,
    "prompt_tokens": 45,
    "completion_tokens": 15
}
```

**Example with Streaming:**
```c
void streaming_callback(const char* token, uint32_t token_id, void* user_data) {
    printf("%s", token);
    fflush(stdout);
}

const char* messages = "[{\"role\": \"user\", \"content\": \"Tell me a story\"}]";

char response[8192];
int result = cactus_complete(model, messages, response, sizeof(response),
                             NULL, NULL, streaming_callback, NULL);
```

### `cactus_tokenize`
Tokenizes text into token IDs using the model's tokenizer.

```c
int cactus_tokenize(
    cactus_model_t model,        // Model handle
    const char* text,            // Text to tokenize
    uint32_t* token_buffer,      // Buffer for token IDs
    size_t token_buffer_len,     // Maximum number of tokens buffer can hold
    size_t* out_token_len        // Output: actual number of tokens
);
```

**Returns:** 0 on success, negative value on error

**Example:**
```c
const char* text = "Hello, world!";
uint32_t tokens[256];
size_t num_tokens = 0;

int result = cactus_tokenize(model, text, tokens, 256, &num_tokens);
if (result == 0) {
    printf("Tokenized into %zu tokens: ", num_tokens);
    for (size_t i = 0; i < num_tokens; i++) {
        printf("%u ", tokens[i]);
    }
    printf("\n");
}
```

### `cactus_score_window`
Scores a window of tokens for perplexity calculation or token probability analysis.

```c
int cactus_score_window(
    cactus_model_t model,        // Model handle
    const uint32_t* tokens,      // Array of token IDs
    size_t token_len,            // Total number of tokens
    size_t start,                // Start index of window to score
    size_t end,                  // End index of window to score
    size_t context,              // Context window size
    char* response_buffer,       // Buffer for response JSON
    size_t buffer_size           // Size of response buffer
);
```

**Returns:** Number of bytes written to response_buffer on success, negative value on error

**Example:**
```c
// First tokenize the text
uint32_t tokens[256];
size_t num_tokens;
cactus_tokenize(model, "The quick brown fox", tokens, 256, &num_tokens);

// Score a window of tokens
char response[4096];
int result = cactus_score_window(model, tokens, num_tokens, 0, num_tokens, 512,
                                  response, sizeof(response));
if (result > 0) {
    printf("Scores: %s\n", response);
}
```

### `cactus_transcribe`
Transcribes audio to text using a Whisper model. Supports both file-based and buffer-based audio input.

```c
int cactus_transcribe(
    cactus_model_t model,           // Model handle (must be Whisper model)
    const char* audio_file_path,    // Path to audio file (WAV, MP3, etc.) - can be NULL if using pcm_buffer
    const char* prompt,             // Optional prompt to guide transcription (can be NULL)
    char* response_buffer,          // Buffer for response JSON
    size_t buffer_size,             // Size of response buffer
    const char* options_json,       // Optional transcription options (can be NULL)
    cactus_token_callback callback, // Optional streaming callback (can be NULL)
    void* user_data,                // User data for callback (can be NULL)
    const uint8_t* pcm_buffer,      // Optional raw PCM audio buffer (can be NULL if using file)
    size_t pcm_buffer_size          // Size of PCM buffer in bytes
);
```

**Returns:** Number of bytes written to response_buffer on success, negative value on error

**Example (file-based):**
```c
cactus_model_t whisper = cactus_init("../../weights/whisper-small", 448, NULL);

char response[16384];
int result = cactus_transcribe(whisper, "audio.wav", NULL,
                                response, sizeof(response), NULL, NULL, NULL,
                                NULL, 0);  // No PCM buffer
if (result > 0) {
    printf("Transcription: %s\n", response);
}
```

**Example (buffer-based):**
```c
// Load PCM audio data (16kHz, mono, 16-bit)
uint8_t* pcm_data = load_audio_buffer("audio.wav", &pcm_size);

char response[16384];
int result = cactus_transcribe(whisper, NULL, NULL,
                                response, sizeof(response), NULL, NULL, NULL,
                                pcm_data, pcm_size);
```

### `cactus_stream_transcribe_t`
An opaque pointer type representing a streaming transcription session. Used for real-time audio transcription with incremental confirmation.

```c
typedef void* cactus_stream_transcribe_t;
```

### `cactus_stream_transcribe_init`
Initializes a new streaming transcription session for a transcription model.

```c
cactus_stream_transcribe_t cactus_stream_transcribe_init(
    cactus_model_t model        // Model handle (must be Whisper model)
);
```

**Returns:** Stream handle on success, NULL on failure

**Example:**
```c
cactus_model_t whisper = cactus_init("../../weights/whisper-small", 448, NULL);

cactus_stream_transcribe_t stream = cactus_stream_transcribe_init(whisper);
if (!stream) {
    fprintf(stderr, "Failed to initialize stream: %s\n", cactus_get_last_error());
    return -1;
}
```

### `cactus_stream_transcribe_insert`
Inserts audio samples into the streaming transcription buffer. Audio should be 16-bit PCM, 16kHz, mono.

```c
int cactus_stream_transcribe_insert(
    cactus_stream_transcribe_t stream,  // Stream handle
    const uint8_t* pcm_buffer,          // Raw PCM audio data (16-bit, 16kHz, mono)
    size_t pcm_buffer_size              // Size of PCM buffer in bytes
);
```

**Returns:** 0 on success, negative value on error

**Example:**
```c
// Insert 1 second of audio (16kHz * 2 bytes per sample)
uint8_t audio_chunk[32000];

int result = cactus_stream_transcribe_insert(stream, audio_chunk, sizeof(audio_chunk));
if (result < 0) {
    fprintf(stderr, "Insert failed: %s\n", cactus_get_last_error());
}
```

### `cactus_stream_transcribe_process`
Processes the accumulated audio buffer and returns confirmed and pending transcription results.

```c
int cactus_stream_transcribe_process(
    cactus_stream_transcribe_t stream,  // Stream handle
    char* response_buffer,              // Buffer for response JSON
    size_t buffer_size,                 // Size of response buffer
    const char* options_json            // Optional processing options (can be NULL)
);
```

**Returns:** Number of bytes written to response_buffer on success, negative value on error

**Options Format:**
```json
{
    "confirmation_threshold": 0.95
}
```

- `confirmation_threshold`: Threshold (0.0-1.0) for confirming transcription segments. Higher values require more stability before confirmation. Default: 0.95

**Response Format:**
```json
{
    "success": true,
    "confirmed": "text confirmed from previous call",
    "pending": "current transcription result"
}
```

- `confirmed`: Text that was confirmed from the previous call (append to final transcription)
- `pending`: Current transcription result (may be confirmed in next call if stable)

**Example:**
```c
char response[32768];
int result = cactus_stream_transcribe_process(stream, response, sizeof(response), "{\"confirmation_threshold\": 0.90}");

if (result > 0) {
    printf("Response: %s\n", response);
}
```

### `cactus_stream_transcribe_finalize`
Finalizes the streaming session and confirms any remaining transcription.

```c
int cactus_stream_transcribe_finalize(
    cactus_stream_transcribe_t stream,  // Stream handle
    char* response_buffer,              // Buffer for response JSON
    size_t buffer_size                  // Size of response buffer
);
```

**Returns:** Number of bytes written to response_buffer on success, negative value on error

**Response Format:**
```json
{
    "success": true,
    "confirmed": "Final confirmed transcription segment"
}
```

**Example:**
```c
// After processing audio chunks
char final_response[32768];
int result = cactus_stream_transcribe_finalize(stream, final_response, sizeof(final_response));

if (result > 0) {
    printf("Final: %s\n", final_response);
}
```

### `cactus_stream_transcribe_destroy`
Releases all resources associated with the streaming transcription session.

```c
void cactus_stream_transcribe_destroy(
    cactus_stream_transcribe_t stream   // Stream handle
);
```

**Example:**
```c
cactus_stream_transcribe_destroy(stream);
```

### `cactus_embed`
Generates text embeddings for semantic search, similarity, and RAG applications.

```c
int cactus_embed(
    cactus_model_t model,        // Model handle
    const char* text,            // Text to embed
    float* embeddings_buffer,    // Buffer for embedding vector
    size_t buffer_size,          // Buffer size in bytes
    size_t* embedding_dim,       // Output: actual embedding dimensions
    bool normalize               // Whether to L2-normalize the output vector
);
```

**Returns:** 0 on success, negative value on error

**Example:**
```c
const char* text = "The quick brown fox jumps over the lazy dog";
float embeddings[2048];
size_t actual_dim = 0;

int result = cactus_embed(model, text, embeddings, sizeof(embeddings), &actual_dim, true);
if (result == 0) {
    printf("Generated %zu-dimensional embedding\n", actual_dim);
}
```

**Note:** Set `normalize` to `true` for cosine similarity comparisons (recommended for most use cases).

### `cactus_image_embed`
Generates embeddings for images, useful for multimodal retrieval tasks.

```c
int cactus_image_embed(
    cactus_model_t model,        // Model handle (must support vision)
    const char* image_path,      // Path to image file
    float* embeddings_buffer,    // Buffer for embedding vector
    size_t buffer_size,          // Buffer size in bytes
    size_t* embedding_dim        // Output: actual embedding dimensions
);
```

**Returns:** 0 on success, negative value on error

**Example:**
```c
float image_embeddings[1024];
size_t dim = 0;

int result = cactus_image_embed(model, "photo.jpg", image_embeddings,
                                 sizeof(image_embeddings), &dim);
if (result == 0) {
    printf("Image embedding dimension: %zu\n", dim);
}
```

### `cactus_audio_embed`
Generates embeddings for audio files, useful for audio retrieval and classification.

```c
int cactus_audio_embed(
    cactus_model_t model,        // Model handle (must support audio)
    const char* audio_path,      // Path to audio file
    float* embeddings_buffer,    // Buffer for embedding vector
    size_t buffer_size,          // Buffer size in bytes
    size_t* embedding_dim        // Output: actual embedding dimensions
);
```

**Returns:** 0 on success, negative value on error

**Example:**
```c
float audio_embeddings[768];
size_t dim = 0;

int result = cactus_audio_embed(model, "speech.wav", audio_embeddings,
                                 sizeof(audio_embeddings), &dim);
```

### `cactus_stop`
Stops ongoing generation. Useful for implementing early stopping based on custom logic.

```c
void cactus_stop(cactus_model_t model);
```

**Example with Controlled Generation:**
```c
struct ControlData {
    cactus_model_t model;
    int token_count;
    int max_tokens;
};

void control_callback(const char* token, uint32_t token_id, void* user_data) {
    struct ControlData* data = (struct ControlData*)user_data;
    printf("%s", token);
    data->token_count++;

    // Stop after reaching limit
    if (data->token_count >= data->max_tokens) {
        cactus_stop(data->model);
    }
}

struct ControlData control = {model, 0, 50};
cactus_complete(model, messages, response, sizeof(response),
                NULL, NULL, control_callback, &control);
```

### `cactus_reset`
Resets the model's internal state, clearing KV cache and any cached context.

```c
void cactus_reset(cactus_model_t model);
```

**Use Cases:**
- Starting a new conversation
- Clearing context between unrelated requests
- Recovering from errors
- Freeing memory after long conversations

### `cactus_destroy`
Releases all resources associated with the model.

```c
void cactus_destroy(cactus_model_t model);
```

**Important:** Always call this when done with a model to prevent memory leaks.

## Utility Functions

### `cactus_get_last_error`
Returns the last error message from the Cactus engine.

```c
const char* cactus_get_last_error(void);
```

**Returns:** Error message string, or NULL if no error

**Example:**
```c
cactus_model_t model = cactus_init("invalid/path", 2048, NULL);
if (!model) {
    const char* error = cactus_get_last_error();
    fprintf(stderr, "Error: %s\n", error);
}
```

### `cactus_set_telemetry_token`
Sets the telemetry token for usage tracking. Pass NULL or empty string to disable telemetry.

```c
void cactus_set_telemetry_token(const char* token);
```

**Example:**
```c
cactus_set_telemetry_token("your-telemetry-token");

cactus_set_telemetry_token(NULL);
```

### `cactus_set_pro_key`
Sets the pro key to enable NPU acceleration on supported devices (Apple Neural Engine).

```c
void cactus_set_pro_key(const char* pro_key);
```

**Example:**
```c
cactus_set_pro_key("your-pro-key");

cactus_model_t model = cactus_init("path/to/model", 2048, NULL);
```

**Note:** The pro key should be set before initializing any models to ensure NPU acceleration is enabled.

## Vector Index APIs

The vector index APIs provide persistent storage and retrieval of embeddings for RAG (Retrieval-Augmented Generation) applications.

### `cactus_index_t`
An opaque pointer type representing a vector index instance.

```c
typedef void* cactus_index_t;
```

### `cactus_index_init`
Initializes or opens a vector index from disk.

```c
cactus_index_t cactus_index_init(
    const char* index_dir,       // Path to index directory
    size_t embedding_dim         // Dimension of embeddings to store
);
```

**Returns:** Index handle on success, NULL on failure

**Example:**
```c
// Create or open an index for 768-dimensional embeddings
cactus_index_t index = cactus_index_init("./my_index", 768);
if (!index) {
    fprintf(stderr, "Failed to initialize index\n");
    return -1;
}
```

### `cactus_index_add`
Adds documents with their embeddings to the index.

```c
int cactus_index_add(
    cactus_index_t index,        // Index handle
    const int* ids,              // Array of document IDs
    const char** documents,      // Array of document texts
    const char** metadatas,      // Array of metadata JSON strings (can be NULL)
    const float** embeddings,    // Array of embedding vectors
    size_t count,                // Number of documents to add
    size_t embedding_dim         // Dimension of each embedding
);
```

**Returns:** 0 on success, negative value on error

**Example:**
```c
// Add documents with embeddings
int ids[] = {1, 2, 3};
const char* docs[] = {"Hello world", "Foo bar", "Test document"};
const char* metas[] = {"{\"source\":\"a\"}", "{\"source\":\"b\"}", NULL};

float emb1[768], emb2[768], emb3[768];
// ... populate embeddings using cactus_embed() ...
const float* embeddings[] = {emb1, emb2, emb3};

int result = cactus_index_add(index, ids, docs, metas, embeddings, 3, 768);
```

### `cactus_index_delete`
Deletes documents from the index by ID.

```c
int cactus_index_delete(
    cactus_index_t index,        // Index handle
    const int* ids,              // Array of document IDs to delete
    size_t ids_count             // Number of IDs
);
```

**Returns:** 0 on success, negative value on error

**Example:**
```c
int ids_to_delete[] = {1, 3};
cactus_index_delete(index, ids_to_delete, 2);
```

### `cactus_index_get`
Retrieves documents by their IDs.

```c
int cactus_index_get(
    cactus_index_t index,        // Index handle
    const int* ids,              // Array of document IDs to retrieve
    size_t ids_count,            // Number of IDs
    char** document_buffers,     // Output: document text buffers
    size_t* document_buffer_sizes,  // Sizes of document buffers
    char** metadata_buffers,     // Output: metadata JSON buffers
    size_t* metadata_buffer_sizes,  // Sizes of metadata buffers
    float** embedding_buffers,   // Output: embedding buffers
    size_t* embedding_buffer_sizes  // Sizes of embedding buffers (in bytes)
);
```

**Returns:** 0 on success, negative value on error

### `cactus_index_query`
Queries the index for similar documents using embedding vectors.

```c
int cactus_index_query(
    cactus_index_t index,        // Index handle
    const float** embeddings,    // Array of query embeddings
    size_t embeddings_count,     // Number of query embeddings
    size_t embedding_dim,        // Dimension of each embedding
    const char* options_json,    // Query options (e.g., {"k": 10})
    int** id_buffers,            // Output: arrays of result IDs
    size_t* id_buffer_sizes,     // Sizes of ID buffers
    float** score_buffers,       // Output: arrays of similarity scores
    size_t* score_buffer_sizes   // Sizes of score buffers
);
```

**Returns:** 0 on success, negative value on error

**Example:**
```c
// Query for similar documents
float query_emb[768];
cactus_embed(model, "search query", query_emb, sizeof(query_emb), &dim, true);

const float* queries[] = {query_emb};
int result_ids[10];
float result_scores[10];
int* id_bufs[] = {result_ids};
float* score_bufs[] = {result_scores};
size_t id_sizes[] = {10};
size_t score_sizes[] = {10};

cactus_index_query(index, queries, 1, 768, "{\"k\": 10}",
                   id_bufs, id_sizes, score_bufs, score_sizes);

for (int i = 0; i < 10; i++) {
    printf("ID: %d, Score: %.4f\n", result_ids[i], result_scores[i]);
}
```

### `cactus_index_compact`
Compacts the index to optimize storage and query performance.

```c
int cactus_index_compact(cactus_index_t index);
```

**Returns:** 0 on success, negative value on error

**Example:**
```c
// Compact after bulk deletions
cactus_index_compact(index);
```

### `cactus_index_destroy`
Releases all resources associated with the index.

```c
void cactus_index_destroy(cactus_index_t index);
```

**Important:** Always call this when done with an index to ensure data is persisted.

### Complete RAG Example

```c
#include "cactus_ffi.h"

int main() {
    // Initialize embedding model
    cactus_model_t embed_model = cactus_init("path/to/embed-model", 512, NULL);

    // Initialize vector index
    cactus_index_t index = cactus_index_init("./rag_index", 768);

    // Add documents
    const char* docs[] = {
        "The capital of France is Paris.",
        "Python is a programming language.",
        "The Earth orbits the Sun."
    };
    int ids[] = {1, 2, 3};
    float emb1[768], emb2[768], emb3[768];
    size_t dim;

    cactus_embed(embed_model, docs[0], emb1, sizeof(emb1), &dim, true);
    cactus_embed(embed_model, docs[1], emb2, sizeof(emb2), &dim, true);
    cactus_embed(embed_model, docs[2], emb3, sizeof(emb3), &dim, true);

    const float* embeddings[] = {emb1, emb2, emb3};
    cactus_index_add(index, ids, docs, NULL, embeddings, 3, 768);

    // Query similar documents
    float query_emb[768];
    cactus_embed(embed_model, "What is the capital of France?", query_emb, sizeof(query_emb), &dim, true);

    const float* queries[] = {query_emb};
    int result_ids[3];
    float result_scores[3];
    int* id_bufs[] = {result_ids};
    float* score_bufs[] = {result_scores};
    size_t id_sizes[] = {3};
    size_t score_sizes[] = {3};

    cactus_index_query(index, queries, 1, 768, "{\"k\": 3}",
                       id_bufs, id_sizes, score_bufs, score_sizes);

    printf("Top result ID: %d (score: %.4f)\n", result_ids[0], result_scores[0]);

    // Cleanup
    cactus_index_destroy(index);
    cactus_destroy(embed_model);
    return 0;
}
```

## Complete Examples

### Basic Conversation
```c
#include "cactus_ffi.h"
#include <stdio.h>
#include <string.h>

int main() {
    // Initialize model
    cactus_model_t model = cactus_init("path/to/model", 2048, NULL);
    if (!model) return -1;

    // Prepare conversation
    const char* messages =
        "[{\"role\": \"system\", \"content\": \"You are a helpful assistant.\"},"
        " {\"role\": \"user\", \"content\": \"Hello!\"},"
        " {\"role\": \"assistant\", \"content\": \"Hello! How can I help you today?\"},"
        " {\"role\": \"user\", \"content\": \"What's 2+2?\"}]";

    // Generate response
    char response[4096];
    int result = cactus_complete(model, messages, response,
                                 sizeof(response), NULL, NULL, NULL, NULL);

    if (result > 0) {
        printf("Response: %s\n", response);
    }

    cactus_destroy(model);
    return 0;
}
```

### Vision-Language Model (VLM)
```c
#include "cactus_ffi.h"

int main() {
    cactus_model_t vlm = cactus_init("path/to/lfm2-vlm", 4096, NULL);
    if (!vlm) return -1;

    // Message with image
    const char* messages =
        "[{\"role\": \"user\","
        "  \"content\": \"What do you see in this image?\","
        "  \"images\": [\"/path/to/photo.jpg\"]}]";

    char response[8192];
    int result = cactus_complete(vlm, messages, response, sizeof(response),
                                 NULL, NULL, NULL, NULL);

    if (result > 0) {
        printf("%s\n", response);
    }

    cactus_destroy(vlm);
    return 0;
}
```

### Tool Calling
```c
const char* tools =
    "[{\"function\": {"
    "    \"name\": \"get_weather\","
    "    \"description\": \"Get weather for a location\","
    "    \"parameters\": {"
    "        \"type\": \"object\","
    "        \"properties\": {"
    "            \"location\": {\"type\": \"string\", \"description\": \"City, State, Country\"}"
    "        },"
    "        \"required\": [\"location\"]"
    "    }"
    "}}]";

const char* messages = "[{\"role\": \"user\", \"content\": \"What's the weather in Paris?\"}]";

char response[4096];
int result = cactus_complete(model, messages, response, sizeof(response),
                             NULL, tools, NULL, NULL);

printf("Response: %s\n", response);
// Parse response JSON to check for function_calls array
```

### Computing Similarity with Embeddings
```c
float compute_cosine_similarity(cactus_model_t model,
                                const char* text1,
                                const char* text2) {
    float embeddings1[2048], embeddings2[2048];
    size_t dim1, dim2;

    // Use normalize=true for cosine similarity (dot product of normalized vectors)
    cactus_embed(model, text1, embeddings1, sizeof(embeddings1), &dim1, true);
    cactus_embed(model, text2, embeddings2, sizeof(embeddings2), &dim2, true);

    // With normalized embeddings, cosine similarity = dot product
    float dot_product = 0.0f;
    for (size_t i = 0; i < dim1; i++) {
        dot_product += embeddings1[i] * embeddings2[i];
    }
    return dot_product;
}

// Usage
float similarity = compute_cosine_similarity(embed_model,
    "The cat sat on the mat",
    "A feline rested on the rug");
printf("Similarity: %.4f\n", similarity);
```

### Audio Transcription with Whisper
```c
#include "cactus_ffi.h"
#include <stdio.h>

void transcription_callback(const char* token, uint32_t token_id, void* user_data) {
    printf("%s", token);
    fflush(stdout);
}

int main() {
    cactus_model_t whisper = cactus_init("path/to/whisper-small", 448, NULL);
    if (!whisper) return -1;

    char response[32768];

    // Transcribe with streaming output
    int result = cactus_transcribe(whisper, "meeting.wav", NULL,
                                    response, sizeof(response), NULL,
                                    transcription_callback, NULL,
                                    NULL, 0);  // No PCM buffer

    printf("\n\nFull response: %s\n", response);

    cactus_destroy(whisper);
    return 0;
}
```

### Multimodal Retrieval
```c
#include "cactus_ffi.h"
#include <math.h>

// Find most similar image to a text query
int find_similar_image(cactus_model_t model,
                       const char* query,
                       const char** image_paths,
                       int num_images) {
    float query_embed[1024];
    size_t query_dim;
    cactus_embed(model, query, query_embed, sizeof(query_embed), &query_dim, true);

    float best_score = -1.0f;
    int best_idx = -1;

    for (int i = 0; i < num_images; i++) {
        float img_embed[1024];
        size_t img_dim;
        cactus_image_embed(model, image_paths[i], img_embed, sizeof(img_embed), &img_dim);

        // Compute cosine similarity
        float dot = 0, norm_q = 0, norm_i = 0;
        for (size_t j = 0; j < query_dim; j++) {
            dot += query_embed[j] * img_embed[j];
            norm_q += query_embed[j] * query_embed[j];
            norm_i += img_embed[j] * img_embed[j];
        }
        float score = dot / (sqrtf(norm_q) * sqrtf(norm_i));

        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    return best_idx;
}
```

## Supported Model Types

| Model Type | Text | Vision | Audio | Embeddings | Description |
|------------|------|--------|-------|------------|-------------|
| Qwen | ✓ | - | - | ✓ | Qwen/Qwen2/Qwen3 language models |
| Gemma | ✓ | - | - | ✓ | Google Gemma models |
| LFM2 | ✓ | ✓ | - | ✓ | Liquid Foundation Models |
| Smol | ✓ | - | - | ✓ | SmolLM compact models |
| Nomic | - | - | - | ✓ | Nomic embedding models |
| Whisper | - | - | ✓ | ✓ | OpenAI Whisper transcription |
| Siglip2 | - | ✓ | - | ✓ | Vision encoder for embeddings |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CACTUS_KV_WINDOW_SIZE` | 512 | Sliding window size for KV cache |
| `CACTUS_KV_SINK_SIZE` | 4 | Number of attention sink tokens to preserve |

**Example:**
```bash
export CACTUS_KV_WINDOW_SIZE=1024
export CACTUS_KV_SINK_SIZE=8
./my_app
```

## Best Practices

1. **Always Check Return Values**: Functions return negative values on error
2. **Buffer Sizes**: Use large response buffers (8192+ bytes recommended)
3. **Memory Management**: Always call `cactus_destroy()` when done
4. **Thread Safety**: Each model instance should be used from a single thread
5. **Context Management**: Use `cactus_reset()` between unrelated conversations
6. **Streaming**: Implement callbacks for better user experience with long generations
7. **Reuse Models**: Initialize once, use multiple times for efficiency

## Error Handling

Most functions return:
- Positive values or 0 on success
- Negative values on error

Common error scenarios:
- Invalid model path
- Insufficient buffer size
- Malformed JSON input
- Unsupported operation for model type
- Out of memory

## Performance Tips

1. **Reuse Model Instances**: Initialize once, use multiple times
2. **Appropriate Context Size**: Use the minimum context size needed for your use case
3. **Streaming for UX**: Use callbacks for responsive user interfaces
4. **Early Stopping**: Use `cactus_stop()` to avoid unnecessary generation
5. **Batch Embeddings**: When possible, process multiple texts in sequence without resetting
6. **KV Cache Tuning**: Adjust `CACTUS_KV_WINDOW_SIZE` based on your context needs