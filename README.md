<img src="assets/banner.jpg" alt="Logo" style="border-radius: 30px; width: 100%;">

```
┌─────────────────┐     Energy-efficient inference engine for running AI on mobile devices 
│  Cactus Engine  │ ←── OpenAI compatible APIs for C/C++, Swift, Kotlin, Flutter & React-Native
└─────────────────┘     Supports tool call, auto RAG, NPU, INT4, and cloud handoff for complex tasks
         │
┌─────────────────┐     Zero-copy computation graph, think PyTorch for mobile devices
│  Cactus Graph   │ ←── You can implement custom models directly using this
└─────────────────┘     Highly optimised for RAM & lossless weight quantisation 
         │
┌─────────────────┐     Low-level ARM-specific SIMD kernels (Apple, Snapdragon, Google, Exynos, MediaTek & Raspberry Pi)
│ Cactus Kernels  │ ←── Optimised Matrix Multiplication & n
└─────────────────┘     Custom attention kernels with KV-Cache Quantisation, chunked prefill, streaming LLM, etc.
```


## Cactus Engine 

```cpp
#include cactus.h

cactus_set_pro_key("optional, email founders@cactuscompute.com"); 

cactus_model_t model = cactus_init(
    "path/to/weight/folder",            
    "path to txt or dir of txts for auto-rag",  
);

const char* messages = R"([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Henry Ndubuaku"}
])";

const char* options = R"({
    "max_tokens": 50,
    "stop_sequences": ["<|im_end|>"]
})";

char response[4096];
int result = cactus_complete(
    model,                            // model handle from cactus_init
    messages,                         // JSON array of chat messages
    response,                         // buffer to store response JSON
    sizeof(response),                 // size of response buffer
    options,                          // optional: generation options (nullptr for defaults)
    nullptr,                          // optional: tools JSON for function calling 
    nullptr,                          // optional: streaming callback fn(token, id, user_data)
    nullptr                           // optional: user data passed to callback
);
```
Example response from Gemma3-270m
```json
{
    "success": true,                 // when successfully generated locally
    "error": null,                   // returns specific errors if success = false
    "cloud_handoff": false,          // true when model is unconfident, simply route to cloud
    "response": "Hi there!",         // null when error is not null or cloud_handoff = true
    "function_calls": [],            // parsed to [{"name":"set_alarm","arguments":{"hour":"10","minute":"0"}}]
    "confidence": 0.8193,            // how confident the model is with its response
    "time_to_first_token_ms": 45.23, // latency (time to first token)
    "total_time_ms": 163.67,         // total execution time
    "prefill_tps": 1621.89,          // prefill tokens per second
    "decode_tps": 168.42,            // decode tokens per second
    "ram_usage_mb": 245.67,          // current process RAM usage in MB
    "prefill_tokens": 28,
    "decode_tokens": 50,
    "total_tokens": 78
}
```

## Cactus Graph

```cpp
#include cactus.h

CactusGraph graph;
auto a = graph.input({2, 3}, Precision::FP16);
auto b = graph.input({3, 4}, Precision::INT8);

auto x1 = graph.matmul(a, b, false);
auto x2 = graph.transpose(x1);
auto result = graph.matmul(b, x2, true);

float a_data[6] = {1.1f, 2.3f, 3.4f, 4.2f, 5.7f, 6.8f};
float b_data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

graph.set_input(a, a_data, Precision::FP16);
graph.set_input(b, b_data, Precision::INT8);

graph.execute();
void* output_data = graph.get_output(result);

graph.hard_reset(); 
```

## High-End Devices Benchmark (INT8)

- Tested for worst case (big model + 1k context size)
- Small models and small context yield flashier numbers, but hides stress points. 

| Device | LFM2.5-1.2B<br>(1k-Prefill/100-Decode) | LFM2.5-VL-1.6B<br>(256px-Latency & Decode) | Whisper-Small<br>(30s-audio-Latency & Decode)
|--------|--------|--------|----------|
| Mac M4 Pro | 582/77 toks/sec| 1.2s(0.3s*) & 76 toks/sec | 1.5s(0.2s*) & 65 toks/sec |
| iPad/Mac M4 | - | - | - |
| iPhone 17 Pro | 300/33 toks/sec | 1.6s(0.3s*) & 33 toks/sec | 3.0s(0.6s*) & 70 toks/sec |
| Galaxy S25 Ultra | 226/35 toks/sec | 2.6s & 35 toks/sec | 2.9s & 44 toks/sec |
| Pixel 10 Pro | - | - | - |
| Vivo X200 Pro | - | - | - |

## Budget Devices Benchmark (INT8)

- We recommend <=600m LLM/VLM and sub-300m transcription for ALL mobiles + cloud fallback
- Cactus decides in sub 100ms when to fallback to private cloud due to complexity, happens <20%

| Device | LFM2-350m<br>(1k-Prefill/100-Decode) | LFM2-VL-450m<br>(256px-Latency & Decode) | Moonshine-Base<br>(30s-audio-Latency & Decode)
|--------|--------|--------|----------|
| iPad/Mac M1 | - | - | - |
| iPhone 13 Mini | - | - | - |
| Galaxy A56 | - | - | - |
| Pixel 6a | 218/44 toks/sec | 3.0s & 42 toks/sec | - |
| Nothing CMF | - | - | - |
| Raspberry Pi 5 | - | - | - |


 ## Supported Models                                                                                                                                                     
                                                                                                                                                                          
  | Model | Size | Features |                                                                                                                                             
  |-------|------|----------|                                                                                                                                             
  | **LLMs** | | |                                                                                                                                                        
  | google/gemma-3-270m-it | 252MB | completion |                                                                                                                         
  | google/functiongemma-270m-it | 252MB | completion, tools |                                                                                                            
  | LiquidAI/LFM2-350M | 244MB | completion, tools, embed |                                                                                                               
  | Qwen/Qwen3-0.6B | 514MB | completion, tools, embed |                                                                                                                  
  | LiquidAI/LFM2-700M | 498MB | completion, tools, embed |                                                                                                               
  | google/gemma-3-1b-it | 642MB | completion |                                                                                                                           
  | LiquidAI/LFM2.5-1.2B-Thinking | 474MB | completion, tools, embed |                                                                                                    
  | LiquidAI/LFM2.5-1.2B-Instruct | 474MB | completion, tools, embed |                                                                                                    
  | LiquidAI/LFM2-1.2B-RAG | 474MB | completion, tools, embed |                                                                                                           
  | LiquidAI/LFM2-1.2B-Tool | 474MB | completion, tools, embed |                                                                                                          
  | Qwen/Qwen3-1.7B | 749MB | completion, tools, embed |                                                                                                                  
  | **VLMs** | | |                                                                                                                                                        
  | LiquidAI/LFM2-VL-450M | 448MB | vision, txt & img embed, Apple NPU |                                                                                                            
  | LiquidAI/LFM2.5-VL-1.6B | 954MB | vision, txt & img embed, Apple NPU |                                                                                                          
  | **Speech** | | |                                                                                                                                                      
  | UsefulSensors/moonshine-base | 80MB | transcription, speech embed |                                                                                                         
  | openai/whisper-small | 283MB | transcription, speech embed, Apple NPU |                                                                                                                 
  | openai/whisper-medium | 658MB | transcription, speech embed, Apple NPU |                                                                                                                
  | **Embeddings** | | |                                                                                                                                                  
  | nomic-ai/nomic-embed-text-v2-moe | 451MB | embed |                                                                                                                    
  | Qwen/Qwen3-Embedding-0.6B | 514MB | embed | 

## Using this repo on a Mac

```bash
git clone https://github.com/cactus-compute/cactus && cd cactus && source ./setup
```

| Command | Description |
|---------|-------------|
| `cactus run [model-name-as-written-in-above-tables]` | Opens playground (auto downloads model) |
| `cactus download [model]` | Downloads model to `./weights` |
| `cactus convert [model] [dir]` | Converts model, supports LoRA merging (`--lora <path>`) |
| `cactus build` | Builds for ARM (`--apple` or `--android`) |
| `cactus test` | Runs tests (`--ios` / `--android`, `--model [name/path]`), `--precision` |
| `cactus clean` | Removes build artifacts |
| `cactus --help` | Shows all commands and flags (please run this to see all commands) |

## Using in your apps 

- [Python for Mac](/python/)
- [React Native SDK](https://github.com/cactus-compute/cactus-react-native)
- [Swift Multiplatform SDK](https://github.com/mhayes853/swift-cactus)
- [Kotlin Multiplatform SDK](https://github.com/cactus-compute/cactus-kotlin)
- [Flutter SDK](https://github.com/cactus-compute/cactus-flutter)
- [Rust SDK](https://github.com/mrsarac/cactus-rs)

## Try demo apps 

- [iOS Demo](https://apps.apple.com/gb/app/cactus-chat/id6744444212)
- [Android Demo](https://play.google.com/store/apps/details?id=com.rshemetsubuser.myapp)

## Maintaining Organisations
1. [Cactus Compute, Inc](https://cactuscompute.com/) 
2. [UCLA's BruinAI](https://bruinai.org/) 
3. [Yale's AI Society](https://www.yale-ai.org/team)
4. [National Unoversity of Singapore's AI Society](https://www.nusaisociety.org/)
5. [UC Irvine's AI@UCI](https://aiclub.ics.uci.edu/)
6. [Imperial College's AI Society](https://www.imperialcollegeunion.org/csp/1391)
7. [University of Pennsylvania's AI@Penn](https://ai-at-penn-main-105.vercel.app/)
8. [Universit of Colorado Boulder's AI Club](https://www.cuaiclub.org/)

## Join The Community
- [Reddit Channel](https://www.reddit.com/r/cactuscompute/)
