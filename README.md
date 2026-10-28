<div align="center">
<pre>
 ██╗  ██╗ ██████╗ ███╗   ███╗██╗
 ██║  ██║██╔═══██╗████╗ ████║██║
 ███████║██║   ██║██╔████╔██║██║
 ██╔══██║██║   ██║██║╚██╔╝██║██║
 ██║  ██║╚██████╔╝██║ ╚═╝ ██║██║
 ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚═╝
 **Your personal AI, connected to your everyday apps.**
</pre>

![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.161+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?style=for-the-badge&logo=mongodb&logoColor=white)
![NVIDIA](https://img.shields.io/badge/NVIDIA-Nemotron-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Mem0](https://img.shields.io/badge/Mem0-Long--Term%20Memory-6C3483?style=for-the-badge)
![Composio](https://img.shields.io/badge/Composio-Tool%20Integrations-FF6B35?style=for-the-badge)
![ElevenLabs](https://img.shields.io/badge/ElevenLabs-TTS-000000?style=for-the-badge)
![Pydantic AI](https://img.shields.io/badge/Pydantic--AI-Agent%20Framework-E92063?style=for-the-badge&logo=pydantic&logoColor=white)

</div>

---

## What is Homi?

**Homi** is a production-ready, multi-user AI agent backend built with a modern Python stack. It acts as a smart personal assistant that remembers who you are across conversations, understands your preferences, and can take real actions on your behalf — like managing your Gmail, Notion, Github through deeply integrated third-party tools.

---

## Capabilities

| Feature | Description |
| :--- | :--- |
| **Conversational AI** | Engages in natural, context-aware conversations with short-term memory per session |
| **Long-Term Memory** | Remembers user facts, preferences, and context across different sessions using Mem0 |
| **User Profiles** | Builds and persists a rich profile per user (name, age, location, custom attributes) in MongoDB |
|  **Tool Integrations** | Connects to real-world apps (Gmail,Notion, Github Calendar, and more) via Composio with OAuth support |
| **Voice-Ready** | ElevenLabs TTS integration ready to convert agent replies to audio for voice interface support |

---

## Architecture

```
                ┌─────────────────────────────────────┐
                │           FastAPI Server             │
                │   POST /chat  DELETE /session        │
                └────────────────┬────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │      Agent Core          │
                    │   (pydantic-ai Agent)    │
                    └─┬──────────┬────────────┘
                      │          │
          ┌───────────▼──┐  ┌────▼──────────────────┐
          │  Mem0 Cloud   │  │     MongoDB Atlas      │
          │  (Long-Term   │  │  ┌─────────────────┐  │
          │   Memories)   │  │  │  user_profiles  │  │
          └───────────────┘  │  ├─────────────────┤  │
                             │  │  chat_sessions  │  │
          ┌────────────────┐ │  │  (TTL: 48hrs)   │  │
          │ Composio MCP   │ │  └─────────────────┘  │
          │ (Gmail, etc.)  │ └───────────────────────┘
          └────────────────┘
```