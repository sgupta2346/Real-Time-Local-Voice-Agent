SAMPLE_RATE = 16000
CHANNELS = 1

OLLAMA_HOST = "http://localhost:11434"
# Non-thinking instruct variant: no <think> reasoning block is generated,
# which matters for a voice agent since every reasoning token is spoken-response
# latency, not just display noise. The plain qwen3:4b tag still emits a full
# reasoning trace even with think:false; this tag genuinely doesn't.
LLM_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
LLM_SYSTEM_PROMPT = (
    "You are a concise voice assistant. Keep replies short (1-3 sentences) "
    "since they will be spoken aloud."
)

WHISPER_MODEL_SIZE = "small"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

KOKORO_VOICE = "af_heart"
