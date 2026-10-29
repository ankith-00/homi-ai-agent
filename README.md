<div align="center">
<pre>
 ██╗  ██╗ ██████╗ ███╗   ███╗██╗
 ██║  ██║██╔═══██╗████╗ ████║██║
 ███████║██║   ██║██╔████╔██║██║
 ██╔══██║██║   ██║██║╚██╔╝██║██║
 ██║  ██║╚██████╔╝██║ ╚═╝ ██║██║
 ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚═╝
 Your personal AI, connected to your everyday apps.
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

```json
[ Introduction ]
{
    "What's Homi ?" :
    "
    Homi is a production-ready, multi-user AI agent backend built with a modern Python stack.
    It acts as a smart personal assistant that remembers who you are across conversations,
    understands your preferences, and can take real actions on your behalf like managing your Gmail,
    Notion, Github through deeply integrated third-party tools.
    ",
    "live_link" : "project-voi.vercel.app/project/homi-agent"
}
```

<br/>

```json
[ Capabilities ]
{
  "conversational_ai": "Natural, context-aware conversations with short-term session memory",
  "long_term_memory": "Remembers user facts, preferences, and context across sessions using Mem0",
  "user_profiles": "Stores user details and custom attributes in MongoDB",
  "tool_integrations": [
    "Gmail",
    "Notion",
    "GitHub",
    "Google Calendar"
  ],
  "integration_provider": "Composio with OAuth support",
  "voice_ready": "ElevenLabs TTS integration for converting agent replies into audio"
}
```

<br>

```text
[ Architecture ]

                 ┌────────────────────────┐
                 │    FastAPI Server      │
                 │  POST /chat            │
                 │  DELETE /session       │
                 └────────────┬───────────┘
                              │
                 ┌────────────▼───────────┐
                 │      Agent Core        │
                 │     Pydantic AI        │
                 └────────────┬───────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
  ┌───────▼────────┐  ┌───────▼────────┐  ┌───────▼────────┐
  │   Mem0 Cloud   │  │ MongoDB Atlas  │  │  Composio MCP  │
  │  Long-term     │  │ user_profiles  │  │ Gmail, Notion  │
  │  memory        │  │ chat_sessions  │  │ GitHub, etc.   │
  └────────────────┘  │ TTL: 48 hours  │  └────────────────┘
                      └────────────────┘
                              │
                 ┌────────────▼───────────┐
                 │   Agent Response       │
                 │  Generated Answer      │
                 └────────────┬───────────┘
                              │
                 ┌────────────▼───────────┐
                 │      Return to User    │
                 │     JSON Response      │
                 └────────────────────────┘
```

<br>
``[ Screenshort - 01 ]``
<img width="1461" height="860" alt="Screenshot 2026-10-29 014023" src="https://github.com/user-attachments/assets/724dfa05-99b1-47f9-9e3f-e288d29df9e2" />
<br>
<br>
<br>
``[ Screenshort - 02 ]``
<img width="1452" height="861" alt="Screenshot 2026-10-29 014043" src="https://github.com/user-attachments/assets/eed3c816-9299-41ad-b1a7-de9eeb8b21d1" />
