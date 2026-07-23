SAMPLE_RATE = 16000
CHANNELS = 1

OLLAMA_HOST = "http://localhost:11434"
# Non-thinking instruct variant: no <think> reasoning block is generated,
# which matters for a voice agent since every reasoning token is spoken-response
# latency, not just display noise (verified: qwen3:4b + think:false still
# emitted a full reasoning trace inline before the answer; this tag doesn't).
LLM_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
LLM_SYSTEM_PROMPT = (
    "You are a concise voice assistant. Keep replies short (1-3 sentences) "
    "since they will be spoken aloud."
)

WHISPER_MODEL_SIZE = "small"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

KOKORO_VOICE = "af_heart"
