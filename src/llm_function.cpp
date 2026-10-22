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

void EnsureHttpRequestLoaded(Connection &conn);

// Forward declaration
std::string GetLlmConfig(ClientContext &context, const std::string &key);

// Raw (unresolved) llm() parameters, as passed by the caller
struct LlmRawParams {
	std::string provider;
	std::string model;
	std::string api_key;
	std::string base_url;
	std::string account_id;
	double temperature = 0.0;
	int max_tokens = 1024;
	bool json_mode = false;
	std::string json_schema;
	std::string return_type;
	std::string reasoning_effort;
	std::string system_prompt;
};

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

	// Constant named params, kept for per-row resolution
	LlmRawParams raw_params;
	// Lateral join named param columns (name, input column index), in declaration order
	vector<std::pair<std::string, idx_t>> param_columns;
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
		Value secret_val = secret->TryGetValue(Identifier(secret_key), false);
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
	EnsureHttpRequestLoaded(conn);

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

// Apply one named parameter; returns false if the key is unknown. NULL values are ignored.
static bool ApplyLlmParam(LlmRawParams &params, const std::string &key_p, const Value &val) {
	auto key = StringUtil::Lower(key_p);
	if (key == "provider") {
		params.provider = val.IsNull() ? params.provider : val.ToString();
	} else if (key == "model") {
		params.model = val.IsNull() ? params.model : val.ToString();
	} else if (key == "api_key") {
		params.api_key = val.IsNull() ? params.api_key : val.ToString();
	} else if (key == "base_url") {
		params.base_url = val.IsNull() ? params.base_url : val.ToString();
	} else if (key == "account_id") {
		params.account_id = val.IsNull() ? params.account_id : val.ToString();
	} else if (key == "temperature") {
		params.temperature = val.IsNull() ? params.temperature : val.DefaultCastAs(LogicalType::DOUBLE).GetValue<double>();
	} else if (key == "max_tokens") {
		params.max_tokens = val.IsNull() ? params.max_tokens : val.DefaultCastAs(LogicalType::INTEGER).GetValue<int>();
	} else if (key == "json_mode") {
		params.json_mode = val.IsNull() ? params.json_mode : val.DefaultCastAs(LogicalType::BOOLEAN).GetValue<bool>();
	} else if (key == "json_schema") {
		params.json_schema = val.IsNull() ? params.json_schema : val.ToString();
	} else if (key == "return_type") {
		params.return_type = val.IsNull() ? params.return_type : val.ToString();
	} else if (key == "reasoning_effort") {
		params.reasoning_effort = val.IsNull() ? params.reasoning_effort : val.ToString();
	} else if (key == "system_prompt") {
		params.system_prompt = val.IsNull() ? params.system_prompt : val.ToString();
	} else {
		return false;
	}
	return true;
}

// Resolve raw params into a request config: param > runtime config > secret > env > default
static void ResolveLlmParams(ClientContext &context, const LlmRawParams &params, LlmBindData &out) {
	out.temperature = params.temperature;
	out.max_tokens = params.max_tokens;
	out.json_mode = params.json_mode;
	out.json_schema = params.json_schema;
	out.return_type = params.return_type;
	out.reasoning_effort = params.reasoning_effort;
	out.system_prompt = params.system_prompt;

	std::string provider_str = params.provider;
	if (provider_str.empty()) {
		provider_str = GetLlmConfig(context, "llm_default_provider");
	}
	if (provider_str.empty()) {
		provider_str = "local";
	}
	out.provider = ParseProvider(provider_str);

	// Try to find a matching secret for this provider
	const KeyValueSecret *secret = nullptr;
	auto &secret_manager = SecretManager::Get(context);
	auto transaction = CatalogTransaction::GetSystemCatalogTransaction(context);
	auto secret_match = secret_manager.LookupSecret(transaction, ProviderToString(out.provider), "llm");
	if (secret_match.HasMatch()) {
		secret = dynamic_cast<const KeyValueSecret *>(secret_match.secret_entry->secret.get());
	}

	switch (out.provider) {
	case LlmProvider::LOCAL:
		out.base_url =
		    ResolveConfig(context, params.base_url, "llm_base_url", "", ProviderDefaults::LOCAL_URL, secret, "base_url");
		out.model = ResolveConfig(context, params.model, "llm_default_model", "", ProviderDefaults::LOCAL_MODEL, secret,
		                          "model");
		out.api_key = params.api_key;
		break;

	case LlmProvider::OPENAI:
		out.base_url = ResolveConfig(context, params.base_url, "llm_openai_base_url", "", ProviderDefaults::OPENAI_URL,
		                             secret, "base_url");
		out.api_key =
		    ResolveConfig(context, params.api_key, "llm_openai_api_key", "OPENAI_API_KEY", "", secret, "api_key");
		out.model = ResolveConfig(context, params.model, "llm_default_model", "", ProviderDefaults::OPENAI_MODEL, secret,
		                          "model");
		if (out.api_key.empty()) {
			throw InvalidInputException("OpenAI API key required. Set OPENAI_API_KEY env, use api_key parameter, "
			                            "SET llm_openai_api_key='...', or CREATE SECRET");
		}
		break;

	case LlmProvider::GEMINI:
		out.base_url = ResolveConfig(context, params.base_url, "", "", ProviderDefaults::GEMINI_URL, secret, "base_url");
		out.api_key =
		    ResolveConfig(context, params.api_key, "llm_gemini_api_key", "GEMINI_API_KEY", "", secret, "api_key");
		out.model = ResolveConfig(context, params.model, "llm_default_model", "", ProviderDefaults::GEMINI_MODEL, secret,
		                          "model");
		if (out.api_key.empty()) {
			throw InvalidInputException("Gemini API key required. Set GEMINI_API_KEY env, use api_key parameter, "
			                            "SET llm_gemini_api_key='...', or CREATE SECRET");
		}
		break;

	case LlmProvider::CLOUDFLARE:
		out.base_url =
		    ResolveConfig(context, params.base_url, "", "", ProviderDefaults::CLOUDFLARE_URL, secret, "base_url");
		out.api_key = ResolveConfig(context, params.api_key, "llm_cloudflare_api_key", "CLOUDFLARE_API_TOKEN", "",
		                            secret, "api_key");
		out.account_id = ResolveConfig(context, params.account_id, "llm_cloudflare_account_id",
		                               "CLOUDFLARE_ACCOUNT_ID", "", secret, "account_id");
		out.model = ResolveConfig(context, params.model, "llm_default_model", "", ProviderDefaults::CLOUDFLARE_MODEL,
		                          secret, "model");
		if (out.api_key.empty()) {
			throw InvalidInputException(
			    "Cloudflare API token required. Set CLOUDFLARE_API_TOKEN env or use api_key parameter");
		}
		if (out.account_id.empty()) {
			throw InvalidInputException(
			    "Cloudflare account_id required. Set CLOUDFLARE_ACCOUNT_ID env or use account_id parameter");
		}
		break;
	}

	// If return_type specified, build JSON schema (provider-specific)
	if (!out.return_type.empty() && out.json_schema.empty()) {
		if (out.provider == LlmProvider::GEMINI) {
			out.json_schema = BuildGeminiReturnTypeSchema(out.return_type);
		} else {
			out.json_schema = BuildOpenAIReturnTypeSchema(out.return_type);
		}
	}
	// Enable json_mode when json_schema is provided
	if (!out.json_schema.empty()) {
		out.json_mode = true;
	}
}

// Bind function
static unique_ptr<FunctionData> LlmBind(ClientContext &context, TableFunctionBindInput &input,
                                        vector<LogicalType> &return_types, vector<Identifier> &names) {
	auto bind_data = make_uniq<LlmBindData>();

	// Parse constant named parameters
	for (auto &kv : input.named_parameters) {
		ApplyLlmParam(bind_data->raw_params, kv.first.GetIdentifierName(), kv.second);
	}

	if (!input.inputs.empty()) {
		auto input_type = input.inputs[0].type();
		if (input_type.id() == LogicalTypeId::VARCHAR) {
			// Static prompt (constant value)
			bind_data->prompt = input.inputs[0].GetValue<std::string>();
			bind_data->prompt_from_input = false;
		} else {
			bind_data->prompt_from_input = true;
		}
	} else if (!input.input_table_types.empty()) {
		// Lateral join: prompt (column 0) and params come from the input DataChunk
		// input_table_names contains aliases like ["", "provider", "model", ...]
		bind_data->prompt_from_input = true;
		for (idx_t i = 1; i < input.input_table_names.size(); i++) {
			auto key = StringUtil::Lower(input.input_table_names[i].GetIdentifierName());
			if (!key.empty()) {
				bind_data->param_columns.emplace_back(key, i);
			}
		}
	} else {
		throw InvalidInputException("llm() requires at least a prompt argument");
	}

	// Without per-row params the config is resolved once here (also validates keys early)
	if (bind_data->param_columns.empty()) {
		ResolveLlmParams(context, bind_data->raw_params, *bind_data);
	}

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

// Table-in-out function - processes input rows and produces output rows
// This enables lateral join support: SELECT t.*, l.* FROM my_table t, llm(t.content) l
static OperatorResultType LlmInOutFunction(ExecutionContext &context, TableFunctionInput &data, DataChunk &input,
                                           DataChunk &output) {
	auto &bind_data = data.bind_data->Cast<LlmBindData>();
	auto &client_context = context.client;

	idx_t output_idx = 0;
	for (idx_t row = 0; row < input.size(); row++) {
		auto prompt_val = input.GetValue(0, row);
		if (prompt_val.IsNull()) {
			output.SetValue(0, output_idx, Value());
			output.SetValue(1, output_idx, Value("error"));
			output_idx++;
			continue;
		}
		std::pair<std::string, std::string> result;
		if (bind_data.param_columns.empty()) {
			result = ExecuteLlmRequest(client_context, prompt_val.ToString(), bind_data);
		} else {
			// Per-row params: overlay row values on the constant params, then resolve
			try {
				LlmRawParams params = bind_data.raw_params;
				for (auto &param : bind_data.param_columns) {
					ApplyLlmParam(params, param.first, input.GetValue(param.second, row));
				}
				LlmBindData row_config;
				ResolveLlmParams(client_context, params, row_config);
				result = ExecuteLlmRequest(client_context, prompt_val.ToString(), row_config);
			} catch (std::exception &ex) {
				ErrorData error(ex);
				result = {error.RawMessage(), "error"};
			}
		}
		output.SetValue(0, output_idx, Value(result.first));
		output.SetValue(1, output_idx, Value(result.second));
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
	llm_func.GetSignature().AddArgs("args", LogicalType::ANY).WithTypedKwargs("options", [](TypedKwargs &options) {
		options.Add("provider", LogicalType::VARCHAR);
		options.Add("model", LogicalType::VARCHAR);
		options.Add("api_key", LogicalType::VARCHAR);
		options.Add("base_url", LogicalType::VARCHAR);
		options.Add("account_id", LogicalType::VARCHAR);
		options.Add("temperature", LogicalType::DOUBLE);
		options.Add("max_tokens", LogicalType::INTEGER);
		options.Add("json_mode", LogicalType::BOOLEAN);
		options.Add("json_schema", LogicalType::VARCHAR);
		options.Add("return_type", LogicalType::VARCHAR);
		options.Add("reasoning_effort", LogicalType::VARCHAR);
		options.Add("system_prompt", LogicalType::VARCHAR);
	});

	loader.RegisterFunction(llm_func);
}

} // namespace duckdb
