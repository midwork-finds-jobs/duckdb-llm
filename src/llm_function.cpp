#include "llm_extension.hpp"
#include "providers.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database_manager.hpp"
#include "duckdb/main/secret/secret.hpp"
#include "duckdb/main/secret/secret_manager.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/blob.hpp"

namespace duckdb {

// Forward declaration
std::string GetLlmConfig(ClientContext &context, const std::string &key);

// Bind data for llm() table function
struct LlmBindData : public TableFunctionData {
	std::string prompt;             // Static prompt (for non-lateral calls)
	bool prompt_from_input = false; // True when prompt comes from input column
	LlmProvider provider = LlmProvider::LOCAL;
	std::string model;
	std::string api_key;
	std::string base_url;
	std::string account_id; // Cloudflare
	double temperature = 0.0;
	int max_tokens = 1024;
	bool json_mode = false;
	std::string json_schema;
	std::string return_type;
	std::string reasoning_effort;
	std::string system_prompt;

	// Column indices for lateral join named params (extracted at execution)
	unordered_map<std::string, idx_t> param_column_indices;
};

// Global state for llm() table function
struct LlmGlobalState : public GlobalTableFunctionState {
	std::string response;
	std::string status;
	bool done = false;
};

// Local state for per-thread execution
struct LlmLocalState : public LocalTableFunctionState {
	idx_t current_row = 0;
};

// Helper to get env var
static std::string GetEnvVar(const char *name) {
	const char *val = getenv(name);
	return val ? std::string(val) : "";
}

// Resolve config value with priority: param > runtime config > secret > env > default
static std::string ResolveConfig(ClientContext &context, const std::string &param_value, const std::string &config_key,
                                 const std::string &env_var, const std::string &default_value,
                                 const KeyValueSecret *secret = nullptr, const std::string &secret_key = "") {
	// 1. Named parameter
	if (!param_value.empty()) {
		return param_value;
	}

	// 2. Runtime config
	std::string config_val = GetLlmConfig(context, config_key);
	if (!config_val.empty()) {
		return config_val;
	}

	// 3. Secret
	if (secret && !secret_key.empty()) {
		Value secret_val = secret->TryGetValue(secret_key, false);
		if (!secret_val.IsNull()) {
			return secret_val.ToString();
		}
	}

	// 4. Environment variable
	if (!env_var.empty()) {
		std::string env_val = GetEnvVar(env_var.c_str());
		if (!env_val.empty()) {
			return env_val;
		}
	}

	// 5. Default
	return default_value;
}

// Build return type JSON schema for OpenAI
static std::string BuildOpenAIReturnTypeSchema(const std::string &return_type) {
	if (return_type.empty()) {
		return "";
	}

	std::string json_type;
	std::string upper_type = StringUtil::Upper(return_type);

	if (upper_type == "INTEGER" || upper_type == "INT" || upper_type == "BIGINT") {
		json_type = "integer";
	} else if (upper_type == "DOUBLE" || upper_type == "FLOAT" || upper_type == "REAL") {
		json_type = "number";
	} else if (upper_type == "BOOLEAN" || upper_type == "BOOL") {
		json_type = "boolean";
	} else if (upper_type == "VARCHAR" || upper_type == "STRING" || upper_type == "TEXT") {
		json_type = "string";
	} else if (upper_type.find("[]") != std::string::npos || upper_type == "ARRAY") {
		// Array type - extract inner type if specified (e.g., "INTEGER[]")
		std::string inner = "string";
		if (upper_type.find("INTEGER") != std::string::npos || upper_type.find("INT") != std::string::npos) {
			inner = "integer";
		} else if (upper_type.find("DOUBLE") != std::string::npos || upper_type.find("FLOAT") != std::string::npos) {
			inner = "number";
		} else if (upper_type.find("BOOLEAN") != std::string::npos || upper_type.find("BOOL") != std::string::npos) {
			inner = "boolean";
		}
		return R"({"name":"result","schema":{"type":"object","properties":{"value":{"type":"array","items":{"type":")" +
		       inner + R"("}}},"required":["value"]},"strict":true})";
	} else {
		throw InvalidInputException(
		    "Unsupported return_type: '%s'. Use INTEGER, DOUBLE, BOOLEAN, VARCHAR, or INTEGER[]", return_type);
	}

	return R"({"name":"result","schema":{"type":"object","properties":{"value":{"type":")" + json_type +
	       R"("}},"required":["value"]},"strict":true})";
}

// Build return type JSON schema for Gemini (uses uppercase types)
static std::string BuildGeminiReturnTypeSchema(const std::string &return_type) {
	if (return_type.empty()) {
		return "";
	}

	std::string json_type;
	std::string upper_type = StringUtil::Upper(return_type);

	if (upper_type == "INTEGER" || upper_type == "INT" || upper_type == "BIGINT") {
		json_type = "INTEGER";
	} else if (upper_type == "DOUBLE" || upper_type == "FLOAT" || upper_type == "REAL") {
		json_type = "NUMBER";
	} else if (upper_type == "BOOLEAN" || upper_type == "BOOL") {
		json_type = "BOOLEAN";
	} else if (upper_type == "VARCHAR" || upper_type == "STRING" || upper_type == "TEXT") {
		json_type = "STRING";
	} else if (upper_type.find("[]") != std::string::npos || upper_type == "ARRAY") {
		// Array type
		std::string inner = "STRING";
		if (upper_type.find("INTEGER") != std::string::npos || upper_type.find("INT") != std::string::npos) {
			inner = "INTEGER";
		} else if (upper_type.find("DOUBLE") != std::string::npos || upper_type.find("FLOAT") != std::string::npos) {
			inner = "NUMBER";
		} else if (upper_type.find("BOOLEAN") != std::string::npos || upper_type.find("BOOL") != std::string::npos) {
			inner = "BOOLEAN";
		}
		return R"({"type":"OBJECT","properties":{"value":{"type":"ARRAY","items":{"type":")" + inner +
		       R"("}}},"required":["value"]})";
	} else {
		throw InvalidInputException(
		    "Unsupported return_type: '%s'. Use INTEGER, DOUBLE, BOOLEAN, VARCHAR, or INTEGER[]", return_type);
	}

	return R"({"type":"OBJECT","properties":{"value":{"type":")" + json_type + R"("}},"required":["value"]})";
}

// Execute LLM request and return response + status
static std::pair<std::string, std::string> ExecuteLlmRequest(ClientContext &context, const std::string &prompt,
                                                             const LlmBindData &bind_data) {
	// Build request body
	std::string request_body;
	std::string url;
	std::string auth_header;

	switch (bind_data.provider) {
	case LlmProvider::LOCAL:
		request_body = BuildLocalRequest(prompt, bind_data.model, bind_data.temperature, bind_data.max_tokens,
		                                 bind_data.json_mode, bind_data.system_prompt);
		url = BuildLocalUrl(bind_data.base_url);
		auth_header = "";
		break;

	case LlmProvider::OPENAI:
		request_body = BuildOpenAIRequest(prompt, bind_data.model, bind_data.temperature, bind_data.max_tokens,
		                                  bind_data.json_mode, bind_data.json_schema, bind_data.system_prompt,
		                                  bind_data.reasoning_effort);
		url = BuildOpenAIUrl(bind_data.base_url);
		auth_header = "Bearer " + bind_data.api_key;
		break;

	case LlmProvider::GEMINI:
		request_body = BuildGeminiRequest(prompt, bind_data.model, bind_data.temperature, bind_data.json_mode,
		                                  bind_data.json_schema, bind_data.system_prompt);
		url = BuildGeminiUrl(bind_data.base_url, bind_data.model, bind_data.api_key);
		auth_header = ""; // Key is in URL
		break;

	case LlmProvider::CLOUDFLARE:
		request_body =
		    BuildCloudflareRequest(prompt, bind_data.temperature, bind_data.max_tokens, bind_data.system_prompt);
		url = BuildCloudflareUrl(bind_data.base_url, bind_data.account_id, bind_data.model);
		auth_header = "Bearer " + bind_data.api_key;
		break;
	}

	// Escape URL for SQL
	std::string escaped_url = StringUtil::Replace(url, "'", "''");

	// Base64 encode the body to avoid any escape sequence issues
	string_t body_str(request_body);
	std::string base64_body = Blob::ToBase64(body_str);

	// Build SQL query to call http_post table function
	std::string query;
	if (auth_header.empty()) {
		query = StringUtil::Format("SELECT status, decode(body) AS body FROM http_post('%s', "
		                           "body := from_base64('%s'), "
		                           "headers := {'Content-Type': 'application/json'})",
		                           escaped_url, base64_body);
	} else {
		std::string escaped_auth = StringUtil::Replace(auth_header, "'", "''");
		query = StringUtil::Format("SELECT status, decode(body) AS body FROM http_post('%s', "
		                           "body := from_base64('%s'), "
		                           "headers := {'Content-Type': 'application/json', 'Authorization': '%s'})",
		                           escaped_url, base64_body, escaped_auth);
	}

	// Execute HTTP request via SQL using a separate connection to avoid deadlock
	auto &db = DatabaseInstance::GetDatabase(context);
	Connection conn(db);

	// Load http_request in this connection
	conn.Query("LOAD http_request");

	auto result = conn.Query(query);

	if (result->HasError()) {
		return {result->GetError(), "error"};
	}

	// Fetch result
	auto chunk = result->Fetch();
	if (!chunk || chunk->size() == 0) {
		return {"No response from HTTP request", "error"};
	}

	// Get status code and body
	auto status_code = chunk->GetValue(0, 0).GetValue<int>();
	auto body = chunk->GetValue(1, 0).GetValue<std::string>();

	if (status_code < 200 || status_code >= 300) {
		return {"HTTP " + std::to_string(status_code) + ": " + body, "error"};
	}

	// Parse response based on provider
	std::string error;
	std::string content;

	switch (bind_data.provider) {
	case LlmProvider::LOCAL:
	case LlmProvider::OPENAI:
		content = ParseOpenAIResponse(body, error);
		break;
	case LlmProvider::GEMINI:
		content = ParseGeminiResponse(body, error);
		break;
	case LlmProvider::CLOUDFLARE:
		content = ParseCloudflareResponse(body, error);
		break;
	}

	if (!error.empty()) {
		return {error, "error"};
	}
	return {content, "ok"};
}

// Bind function
static unique_ptr<FunctionData> LlmBind(ClientContext &context, TableFunctionBindInput &input,
                                        vector<LogicalType> &return_types, vector<string> &names) {
	auto bind_data = make_uniq<LlmBindData>();

	// Parse named parameters (works for standard calls)
	std::string provider_str = "local";
	std::string param_api_key;
	std::string param_base_url;
	std::string param_model;
	std::string param_account_id;

	for (auto &kv : input.named_parameters) {
		auto key = StringUtil::Lower(kv.first);
		if (key == "provider") {
			provider_str = kv.second.GetValue<std::string>();
		} else if (key == "model") {
			param_model = kv.second.GetValue<std::string>();
		} else if (key == "api_key") {
			param_api_key = kv.second.GetValue<std::string>();
		} else if (key == "base_url") {
			param_base_url = kv.second.GetValue<std::string>();
		} else if (key == "account_id") {
			param_account_id = kv.second.GetValue<std::string>();
		} else if (key == "temperature") {
			bind_data->temperature = kv.second.GetValue<double>();
		} else if (key == "max_tokens") {
			bind_data->max_tokens = kv.second.GetValue<int>();
		} else if (key == "json_mode") {
			bind_data->json_mode = kv.second.GetValue<bool>();
		} else if (key == "json_schema") {
			bind_data->json_schema = kv.second.GetValue<std::string>();
		} else if (key == "return_type") {
			bind_data->return_type = kv.second.GetValue<std::string>();
		} else if (key == "reasoning_effort") {
			bind_data->reasoning_effort = kv.second.GetValue<std::string>();
		} else if (key == "system_prompt") {
			bind_data->system_prompt = kv.second.GetValue<std::string>();
		}
	}

	// Determine if prompt is a constant or column reference
	if (!input.inputs.empty()) {
		// Check if this is a constant or column reference
		auto input_type = input.inputs[0].type();
		if (input_type.id() == LogicalTypeId::VARCHAR) {
			// Static prompt (constant value)
			bind_data->prompt = input.inputs[0].GetValue<std::string>();
			bind_data->prompt_from_input = false;
		} else {
			// Column reference will be handled at execution time
			bind_data->prompt_from_input = true;
		}
	} else if (!input.input_table_types.empty()) {
		// Lateral join: prompt and params come from input DataChunk
		// When using in_out_function with column refs, DuckDB converts named params to additional columns
		// input_table_names contains aliases like ["", "provider", "model", ...]
		bind_data->prompt_from_input = true;

		// Store column indices for named parameters (values extracted at execution time)
		for (idx_t i = 1; i < input.input_table_names.size(); i++) {
			auto key = StringUtil::Lower(input.input_table_names[i]);
			if (!key.empty()) {
				bind_data->param_column_indices[key] = i;
			}
		}
	} else {
		throw InvalidInputException("llm() requires at least a prompt argument");
	}

	// Check runtime config for provider if not specified via param
	if (provider_str == "local") {
		std::string config_provider = GetLlmConfig(context, "llm_default_provider");
		if (!config_provider.empty()) {
			provider_str = config_provider;
		}
	}

	// Parse provider
	bind_data->provider = ParseProvider(provider_str);

	// Try to find a matching secret for this provider
	const KeyValueSecret *secret = nullptr;
	auto &secret_manager = SecretManager::Get(context);
	auto transaction = CatalogTransaction::GetSystemCatalogTransaction(context);

	// Look for secret with scope matching provider
	auto secret_match = secret_manager.LookupSecret(transaction, ProviderToString(bind_data->provider), "llm");
	if (secret_match.HasMatch()) {
		secret = dynamic_cast<const KeyValueSecret *>(secret_match.secret_entry->secret.get());
	}

	// Resolve configuration based on provider
	switch (bind_data->provider) {
	case LlmProvider::LOCAL:
		bind_data->base_url =
		    ResolveConfig(context, param_base_url, "llm_base_url", "", ProviderDefaults::LOCAL_URL, secret, "base_url");
		bind_data->model = ResolveConfig(context, param_model, "llm_default_model", "", ProviderDefaults::LOCAL_MODEL,
		                                 secret, "model");
		break;

	case LlmProvider::OPENAI:
		bind_data->base_url = ResolveConfig(context, param_base_url, "llm_openai_base_url", "",
		                                    ProviderDefaults::OPENAI_URL, secret, "base_url");
		bind_data->api_key =
		    ResolveConfig(context, param_api_key, "llm_openai_api_key", "OPENAI_API_KEY", "", secret, "api_key");
		bind_data->model = ResolveConfig(context, param_model, "llm_default_model", "", ProviderDefaults::OPENAI_MODEL,
		                                 secret, "model");
		if (bind_data->api_key.empty()) {
			throw InvalidInputException("OpenAI API key required. Set OPENAI_API_KEY env, use api_key parameter, "
			                            "SET llm_openai_api_key='...', or CREATE SECRET");
		}
		break;

	case LlmProvider::GEMINI:
		bind_data->base_url =
		    ResolveConfig(context, param_base_url, "", "", ProviderDefaults::GEMINI_URL, secret, "base_url");
		bind_data->api_key =
		    ResolveConfig(context, param_api_key, "llm_gemini_api_key", "GEMINI_API_KEY", "", secret, "api_key");
		bind_data->model = ResolveConfig(context, param_model, "llm_default_model", "", ProviderDefaults::GEMINI_MODEL,
		                                 secret, "model");
		if (bind_data->api_key.empty()) {
			throw InvalidInputException("Gemini API key required. Set GEMINI_API_KEY env, use api_key parameter, "
			                            "SET llm_gemini_api_key='...', or CREATE SECRET");
		}
		break;

	case LlmProvider::CLOUDFLARE:
		bind_data->base_url =
		    ResolveConfig(context, param_base_url, "", "", ProviderDefaults::CLOUDFLARE_URL, secret, "base_url");
		bind_data->api_key = ResolveConfig(context, param_api_key, "llm_cloudflare_api_key", "CLOUDFLARE_API_TOKEN", "",
		                                   secret, "api_key");
		bind_data->account_id = ResolveConfig(context, param_account_id, "llm_cloudflare_account_id",
		                                      "CLOUDFLARE_ACCOUNT_ID", "", secret, "account_id");
		bind_data->model = ResolveConfig(context, param_model, "llm_default_model", "",
		                                 ProviderDefaults::CLOUDFLARE_MODEL, secret, "model");
		if (bind_data->api_key.empty()) {
			throw InvalidInputException(
			    "Cloudflare API token required. Set CLOUDFLARE_API_TOKEN env or use api_key parameter");
		}
		if (bind_data->account_id.empty()) {
			throw InvalidInputException(
			    "Cloudflare account_id required. Set CLOUDFLARE_ACCOUNT_ID env or use account_id parameter");
		}
		break;
	}

	// If return_type specified, build JSON schema (provider-specific)
	if (!bind_data->return_type.empty() && bind_data->json_schema.empty()) {
		if (bind_data->provider == LlmProvider::GEMINI) {
			bind_data->json_schema = BuildGeminiReturnTypeSchema(bind_data->return_type);
		} else {
			bind_data->json_schema = BuildOpenAIReturnTypeSchema(bind_data->return_type);
		}
		bind_data->json_mode = true;
	}

	// Enable json_mode when json_schema is provided
	if (!bind_data->json_schema.empty()) {
		bind_data->json_mode = true;
	}

	// Set return types
	names = {"response", "status"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR};

	return std::move(bind_data);
}

// Global init - minimal setup (no HTTP execution here)
static unique_ptr<GlobalTableFunctionState> LlmInitGlobal(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<LlmGlobalState>();
}

// Local init - per-thread state
static unique_ptr<LocalTableFunctionState> LlmInitLocal(ExecutionContext &context, TableFunctionInitInput &input,
                                                        GlobalTableFunctionState *global_state) {
	return make_uniq<LlmLocalState>();
}

// Helper to extract string param from input chunk
static std::string ExtractParamString(DataChunk &input, idx_t col_idx) {
	if (col_idx >= input.ColumnCount()) {
		return "";
	}
	auto val = input.GetValue(col_idx, 0);
	if (val.IsNull()) {
		return "";
	}
	return val.GetValue<std::string>();
}

// Table-in-out function - processes input rows and produces output rows
// This enables lateral join support: SELECT t.*, l.* FROM my_table t, llm(t.content) l
static OperatorResultType LlmInOutFunction(ExecutionContext &context, TableFunctionInput &data, DataChunk &input,
                                           DataChunk &output) {
	auto &orig_bind_data = data.bind_data->Cast<LlmBindData>();
	auto &client_context = context.client;

	if (input.size() == 0) {
		output.SetCardinality(0);
		return OperatorResultType::NEED_MORE_INPUT;
	}

	// Create a copy of bind_data that we can modify with lateral params
	LlmBindData bind_data = orig_bind_data;

	// Extract lateral join named parameters from input columns (constant for all rows)
	std::string new_provider;
	for (auto &kv : orig_bind_data.param_column_indices) {
		const std::string &key = kv.first;
		idx_t col_idx = kv.second;
		std::string val = ExtractParamString(input, col_idx);

		if (key == "provider" && !val.empty()) {
			new_provider = val;
			bind_data.provider = ParseProvider(val);
			// Clear model/base_url so they get resolved for new provider
			bind_data.model = "";
			bind_data.base_url = "";
		} else if (key == "model" && !val.empty()) {
			bind_data.model = val;
		} else if (key == "api_key" && !val.empty()) {
			bind_data.api_key = val;
		} else if (key == "base_url" && !val.empty()) {
			bind_data.base_url = val;
		} else if (key == "account_id" && !val.empty()) {
			bind_data.account_id = val;
		} else if (key == "system_prompt" && !val.empty()) {
			bind_data.system_prompt = val;
		} else if (key == "json_schema" && !val.empty()) {
			bind_data.json_schema = val;
			bind_data.json_mode = true;
		} else if (key == "return_type" && !val.empty()) {
			bind_data.return_type = val;
		} else if (key == "reasoning_effort" && !val.empty()) {
			bind_data.reasoning_effort = val;
		}
		// Note: temperature, max_tokens, json_mode would need type conversion
	}

	// If provider was changed via lateral params, resolve api_key from env if not set
	if (!new_provider.empty() && bind_data.api_key.empty()) {
		switch (bind_data.provider) {
		case LlmProvider::OPENAI:
			bind_data.api_key = GetEnvVar("OPENAI_API_KEY");
			if (bind_data.base_url.empty()) {
				bind_data.base_url = ProviderDefaults::OPENAI_URL;
			}
			if (bind_data.model.empty()) {
				bind_data.model = ProviderDefaults::OPENAI_MODEL;
			}
			break;
		case LlmProvider::GEMINI:
			bind_data.api_key = GetEnvVar("GEMINI_API_KEY");
			if (bind_data.base_url.empty()) {
				bind_data.base_url = ProviderDefaults::GEMINI_URL;
			}
			if (bind_data.model.empty()) {
				bind_data.model = ProviderDefaults::GEMINI_MODEL;
			}
			break;
		case LlmProvider::CLOUDFLARE:
			bind_data.api_key = GetEnvVar("CLOUDFLARE_API_TOKEN");
			if (bind_data.account_id.empty()) {
				bind_data.account_id = GetEnvVar("CLOUDFLARE_ACCOUNT_ID");
			}
			if (bind_data.base_url.empty()) {
				bind_data.base_url = ProviderDefaults::CLOUDFLARE_URL;
			}
			if (bind_data.model.empty()) {
				bind_data.model = ProviderDefaults::CLOUDFLARE_MODEL;
			}
			break;
		default:
			break;
		}
	}

	// Process each row in the input chunk
	idx_t output_idx = 0;
	for (idx_t row = 0; row < input.size(); row++) {
		// Get prompt from input (first column)
		auto prompt = input.GetValue(0, row).GetValue<std::string>();

		// Execute LLM request for this row
		auto result = ExecuteLlmRequest(client_context, prompt, bind_data);
		std::string response = result.first;
		std::string status = result.second;

		// Set output values
		output.SetValue(0, output_idx, Value(response));
		output.SetValue(1, output_idx, Value(status));
		output_idx++;
	}

	output.SetCardinality(output_idx);
	return OperatorResultType::NEED_MORE_INPUT;
}

// Legacy scan function for backwards compatibility (single prompt, no lateral join)
static void LlmScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind_data = data.bind_data->Cast<LlmBindData>();
	auto &global_state = data.global_state->Cast<LlmGlobalState>();

	// Lateral joins use in_out_function, not this scan function
	if (bind_data.prompt_from_input) {
		output.SetCardinality(0);
		return;
	}

	if (global_state.done) {
		output.SetCardinality(0);
		return;
	}

	// Execute the request using the prompt from bind_data
	std::string response, status;
	auto result = ExecuteLlmRequest(context, bind_data.prompt, bind_data);
	response = result.first;
	status = result.second;

	output.SetValue(0, 0, Value(response));
	output.SetValue(1, 0, Value(status));
	output.SetCardinality(1);

	global_state.done = true;
}

void RegisterLlmFunction(ExtensionLoader &loader) {
	// Use LogicalType::ANY to allow both constants and column references (for lateral joins)
	TableFunction llm_func("llm", {LogicalType::ANY}, LlmScan, LlmBind, LlmInitGlobal);

	// Enable table-in-out function for lateral join support
	llm_func.init_local = LlmInitLocal;
	llm_func.in_out_function = LlmInOutFunction;

	// Allow varargs for lateral join named params (they become additional columns)
	llm_func.varargs = LogicalType::ANY;

	// Add named parameters
	llm_func.named_parameters["provider"] = LogicalType::VARCHAR;
	llm_func.named_parameters["model"] = LogicalType::VARCHAR;
	llm_func.named_parameters["api_key"] = LogicalType::VARCHAR;
	llm_func.named_parameters["base_url"] = LogicalType::VARCHAR;
	llm_func.named_parameters["account_id"] = LogicalType::VARCHAR;
	llm_func.named_parameters["temperature"] = LogicalType::DOUBLE;
	llm_func.named_parameters["max_tokens"] = LogicalType::INTEGER;
	llm_func.named_parameters["json_mode"] = LogicalType::BOOLEAN;
	llm_func.named_parameters["json_schema"] = LogicalType::VARCHAR;
	llm_func.named_parameters["return_type"] = LogicalType::VARCHAR;
	llm_func.named_parameters["reasoning_effort"] = LogicalType::VARCHAR;
	llm_func.named_parameters["system_prompt"] = LogicalType::VARCHAR;

	loader.RegisterFunction(llm_func);
}

} // namespace duckdb
