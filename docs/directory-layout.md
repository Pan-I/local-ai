# Main Machine

local-ai/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── app/                    # Application entry points
│   ├── main.py
│   ├── ollama_client.py
│   ├── config.py
│   └── conversation.py
│
├── agent/                  # Agent architecture
│   ├── planner.py
│   ├── executor.py
│   ├── tools.py
│   └── state.py
│
├── storage/                # Persistence layer
│   ├── database.py
│   ├── models.py
│   └── repositories/
│
├── prompts/                # Shared prompt templates
│   ├── system/
│   ├── agents/
│   └── templates/
│
├── projects/               # Individual AI projects
│   ├── project_alpha/
│   │   ├── project.yaml
│   │   ├── prompts/
│   │   ├── knowledge/
│   │   ├── workspace/
│   │   ├── artifacts/
│   │   └── tests/
│   │
│   └── project_beta/
│
├── data/                   # Persistent application data
│   ├── databases/
│   │   └── local_ai.db
│   ├── embeddings/
│   └── indexes/
│
├── config/                 # Machine-specific configuration
│   ├── config.example.yaml
│   └── local.yaml
│
├── logs/
│
├── tests/                  # Application-level tests
│
└── backups/