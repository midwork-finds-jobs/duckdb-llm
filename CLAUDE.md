# DuckDB LLM Extension (C++)

Rewrite of the Rust DuckDB LLM extension in C++ for full DuckDB API access (including CREATE SECRET).

## Goal

Native C++ DuckDB extension for calling LLM APIs directly from SQL.

## Features to Implement

### Core Functions

1. **`llm()` table function** - Full provider/parameter control
2. **`prompt()` scalar function** - Simple inline usage with env auto-detect
3. **`llm_set_config()` / `llm_get_config()` scalar functions** - Runtime config

### Providers

| Provider | Default Model | Default URL | Auth |
|----------|---------------|-------------|------|
| local | default | `http://localhost:1234` | None |
| openai | gpt-4o-mini | `https://api.openai.com` | API key |
| gemini | gemini-2.5-flash-lite | googleapis.com | API key |
| cloudflare | @cf/meta/llama-4-scout-17b-16e-instruct | cloudflare.com | API key + account_id |

### Parameters for llm()

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| prompt | VARCHAR | required | Text to send to model |
| provider | VARCHAR | 'local' | 'local', 'openai', 'gemini', 'cloudflare' |
| model | VARCHAR | auto | Model name |
| api_key | VARCHAR | '' | API key |
| base_url | VARCHAR | auto | Custom API URL |
| account_id | VARCHAR | '' | Cloudflare account ID |
| temperature | DOUBLE | 0.0 | Temperature (0-1) |
| max_tokens | INTEGER | 1024 | Max response tokens |
| json_mode | BOOLEAN | false | Force JSON response |
| json_schema | VARCHAR | NULL | JSON schema for structured output |
| return_type | VARCHAR | NULL | INTEGER, BOOLEAN, VARCHAR, etc. |
| reasoning_effort | VARCHAR | NULL | OpenAI o1/o3: 'low', 'medium', 'high' |
| system_prompt | VARCHAR | NULL | System message |

### Return Columns

| Column | Type | Description |
|--------|------|-------------|
| response | VARCHAR | LLM response or error message |
| status | VARCHAR | 'ok' or 'error' |

## Lateral Join Support

The `llm()` table function supports lateral joins for per-row LLM calls with full named parameter support:

```sql
-- Process each row with llm() and named parameters
SELECT t.id, l.response
FROM my_table t, llm(t.content, provider := 'gemini', model := 'gemini-2.5-flash') l;

-- Or with explicit LATERAL syntax
SELECT t.id, l.response
FROM my_table t CROSS JOIN LATERAL llm(t.content, provider := 'openai') l;
```

**Alternative:** The `prompt()` scalar function auto-detects API keys from environment:

```sql
-- prompt() works per-row, auto-detects GEMINI_API_KEY from env
SELECT t.id, prompt(t.content) as response FROM my_table t;
```

## Config Resolution Priority

1. Named parameter (e.g., `api_key := 'sk-...'`)
2. Runtime config (`llm_set_config()`)
3. Environment variable
4. Default value

### Environment Variables

| Variable | Provider |
|----------|----------|
| OPENAI_API_KEY | openai |
| GEMINI_API_KEY | gemini |
| CLOUDFLARE_API_TOKEN | cloudflare |
| CLOUDFLARE_ACCOUNT_ID | cloudflare |

### Config Keys

| Key | Description |
|-----|-------------|
| base_url | Local provider base URL |
| openai_base_url | OpenAI provider base URL |
| openai_api_key | OpenAI API key |
| gemini_api_key | Gemini API key |
| cloudflare_api_key | Cloudflare API token |
| cloudflare_account_id | Cloudflare account ID |

## CREATE SECRET Support (C++ only)

Implement using DuckDB's SecretManager API:

```cpp
#include "duckdb/main/secret/secret.hpp"
#include "duckdb/main/secret/secret_manager.hpp"

// Register secret type
SecretType llm_secret_type;
llm_secret_type.name = "llm";
llm_secret_type.deserializer = LlmSecretDeserializer;
llm_secret_type.default_provider = "config";
secret_manager.RegisterSecretType(llm_secret_type);

// Register create secret function
CreateSecretFunction create_func;
create_func.secret_type = "llm";
create_func.provider = "openai";  // or gemini, cloudflare
create_func.named_parameters["api_key"] = LogicalType::VARCHAR;
create_func.named_parameters["base_url"] = LogicalType::VARCHAR;
secret_manager.RegisterSecretFunction(create_func, OnCreateConflict::ERROR_ON_CONFLICT);
```

Target SQL syntax:

```sql
CREATE SECRET openai_secret (
    TYPE llm,
    PROVIDER openai,
    api_key 'sk-...',
    base_url 'https://api.openai.com'
);

CREATE SECRET gemini_secret (
    TYPE llm,
    PROVIDER gemini,
    api_key 'AIza...'
);

CREATE SECRET cloudflare_secret (
    TYPE llm,
    PROVIDER cloudflare,
    api_key 'cf-...',
    account_id '...'
);
```

## API Request Formats

### OpenAI / Local (OpenAI-compatible)

```text
POST {base_url}/v1/chat/completions
Authorization: Bearer {api_key}
Content-Type: application/json

{
  "model": "{model}",
  "messages": [
    {"role": "system", "content": "{system_prompt}"},  // optional
    {"role": "user", "content": "{prompt}"}
  ],
  "temperature": {temperature},
  "max_tokens": {max_tokens},
  "response_format": {"type": "json_object"}  // if json_mode
  // OR for structured output:
  "response_format": {
    "type": "json_schema",
    "json_schema": {json_schema}
  }
}
```

Response: `choices[0].message.content`

### Gemini

```text
POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}
Content-Type: application/json

{
  "contents": [{"parts": [{"text": "{prompt}"}]}],
  "systemInstruction": {"parts": [{"text": "{system_prompt}"}]},  // gemini-* only
  "generationConfig": {
    "temperature": {temperature},
    "responseMimeType": "application/json",  // if json_mode
    "responseSchema": {json_schema}  // gemini-* only
  }
}
```

Response: `candidates[0].content.parts[0].text`

**Note:** `gemini-*` models support native structured output. `gemma*` models don't - use system_prompt fallback (prepend to user message since gemma doesn't support systemInstruction).

### Cloudflare Workers AI

```text
POST https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}
Authorization: Bearer {api_key}
Content-Type: application/json

{
  "messages": [
    {"role": "system", "content": "{system_prompt}"},  // optional
    {"role": "user", "content": "{prompt}"}
  ],
  "temperature": {temperature},
  "max_tokens": {max_tokens}
}
```

Response: `result.response`

## Structured Output

### return_type Parameter

When `return_type` is specified, generate JSON schema:

```cpp
std::string return_type_to_schema(const std::string& type) {
    // INTEGER, INT, BIGINT -> {"type": "integer"}
    // DOUBLE, FLOAT -> {"type": "number"}
    // BOOLEAN, BOOL -> {"type": "boolean"}
    // VARCHAR, STRING, TEXT -> {"type": "string"}

    return R"({
        "name": "result",
        "schema": {
            "type": "object",
            "properties": {
                "value": {"type": ")" + json_type + R"("}
            },
            "required": ["value"]
        },
        "strict": true
    })";
}
```

### prompt() Scalar Auto-detect

```cpp
// Priority: GEMINI_API_KEY > OPENAI_API_KEY > local
if (getenv("GEMINI_API_KEY")) {
    provider = "gemini";
    api_key = getenv("GEMINI_API_KEY");
} else if (getenv("OPENAI_API_KEY")) {
    provider = "openai";
    api_key = getenv("OPENAI_API_KEY");
} else {
    provider = "local";
}
```

## HTTP Client

Use DuckDB's built-in HTTP client or link against a library like libcurl or cpp-httplib.

## Build

Use DuckDB extension template:

```bash
git clone --recursive https://github.com/duckdb/extension-template duckdb-llm2
cd duckdb-llm2
./scripts/bootstrap-template.py llm
make
```

## Testing

Create test files in `test/sql/`:

- `test_openai.sql`
- `test_gemini.sql`
- `test_cloudflare.sql`
- `test_local.sql`
- `test_config.sql`
- `test_secrets.sql`
- `test_structured_output.sql`

## Reference Implementation

See Rust implementation in `../duckdb_llm/rust/src/lib.rs` for complete logic.
