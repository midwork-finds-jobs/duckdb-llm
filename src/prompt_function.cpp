#include "llm_extension.hpp"
#include "providers.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/common/exception.hpp"

namespace duckdb {

// Forward declaration
std::string GetLlmConfig(ClientContext &context, const std::string &key);

// Helper to get env var
static std::string GetEnvVar(const char *name) {
	const char *val = getenv(name);
	return val ? std::string(val) : "";
}

// Auto-detect provider from environment
static void DetectProvider(std::string &provider, std::string &api_key, std::string &base_url, std::string &model) {
	// Priority: GEMINI_API_KEY > OPENAI_API_KEY > local
	std::string gemini_key = GetEnvVar("GEMINI_API_KEY");
	std::string openai_key = GetEnvVar("OPENAI_API_KEY");

	if (!gemini_key.empty()) {
		provider = "gemini";
		api_key = gemini_key;
		base_url = ProviderDefaults::GEMINI_URL;
		model = ProviderDefaults::GEMINI_MODEL;
	} else if (!openai_key.empty()) {
		provider = "openai";
		api_key = openai_key;
		base_url = ProviderDefaults::OPENAI_URL;
		model = ProviderDefaults::OPENAI_MODEL;
	} else {
		provider = "local";
		api_key = "";
		base_url = ProviderDefaults::LOCAL_URL;
		model = ProviderDefaults::LOCAL_MODEL;
	}
}

// Execute LLM call and return response
static std::string ExecutePrompt(ClientContext &context, const std::string &prompt) {
	std::string provider, api_key, base_url, model;
	DetectProvider(provider, api_key, base_url, model);

	// Check runtime config for overrides
	std::string config_provider = GetLlmConfig(context, "llm_default_provider");
	if (!config_provider.empty() && config_provider != "local") {
		provider = config_provider;
		auto prov = ParseProvider(provider);
		switch (prov) {
		case LlmProvider::LOCAL:
			base_url = ProviderDefaults::LOCAL_URL;
			model = ProviderDefaults::LOCAL_MODEL;
			break;
		case LlmProvider::OPENAI:
			base_url = ProviderDefaults::OPENAI_URL;
			model = ProviderDefaults::OPENAI_MODEL;
			api_key = GetLlmConfig(context, "llm_openai_api_key");
			if (api_key.empty()) {
				api_key = GetEnvVar("OPENAI_API_KEY");
			}
			break;
		case LlmProvider::GEMINI:
			base_url = ProviderDefaults::GEMINI_URL;
			model = ProviderDefaults::GEMINI_MODEL;
			api_key = GetLlmConfig(context, "llm_gemini_api_key");
			if (api_key.empty()) {
				api_key = GetEnvVar("GEMINI_API_KEY");
			}
			break;
		case LlmProvider::CLOUDFLARE:
			throw InvalidInputException(
			    "prompt() does not support Cloudflare provider (requires account_id). Use llm() instead.");
		}
	}

	// Build request
	std::string request_body;
	std::string url;
	std::string auth_header;
	LlmProvider prov = ParseProvider(provider);

	switch (prov) {
	case LlmProvider::LOCAL:
		request_body = BuildLocalRequest(prompt, model, 0.0, 1024, false, "");
		url = BuildLocalUrl(base_url);
		auth_header = "";
		break;

	case LlmProvider::OPENAI:
		if (api_key.empty()) {
			throw InvalidInputException("OpenAI API key required. Set OPENAI_API_KEY or SET llm_openai_api_key='...'");
		}
		request_body = BuildOpenAIRequest(prompt, model, 0.0, 1024, false, "", "", "");
		url = BuildOpenAIUrl(base_url);
		auth_header = "Bearer " + api_key;
		break;

	case LlmProvider::GEMINI:
		if (api_key.empty()) {
			throw InvalidInputException("Gemini API key required. Set GEMINI_API_KEY or SET llm_gemini_api_key='...'");
		}
		request_body = BuildGeminiRequest(prompt, model, 0.0, false, "", "");
		url = BuildGeminiUrl(base_url, model, api_key);
		auth_header = "";
		break;

	default:
		throw InvalidInputException("Unsupported provider for prompt()");
	}

	// Escape for SQL
	std::string escaped_body = EscapeJsonString(request_body);
	std::string escaped_url = StringUtil::Replace(url, "'", "''");

	// Build SQL query
	std::string query;
	if (auth_header.empty()) {
		query = StringUtil::Format("SELECT status, body FROM (SELECT (http_post('%s', '%s', "
		                           "headers := {'Content-Type': 'application/json'})).*)",
		                           escaped_url, escaped_body);
	} else {
		std::string escaped_auth = StringUtil::Replace(auth_header, "'", "''");
		query = StringUtil::Format("SELECT status, body FROM (SELECT (http_post('%s', '%s', "
		                           "headers := {'Content-Type': 'application/json', 'Authorization': '%s'})).*)",
		                           escaped_url, escaped_body, escaped_auth);
	}

	// Execute
	auto result = context.Query(query, false);
	if (result->HasError()) {
		throw IOException("LLM request failed: " + result->GetError());
	}

	auto chunk = result->Fetch();
	if (!chunk || chunk->size() == 0) {
		throw IOException("No response from LLM");
	}

	auto status_code = chunk->GetValue(0, 0).GetValue<int>();
	auto body = chunk->GetValue(1, 0).GetValue<std::string>();

	if (status_code < 200 || status_code >= 300) {
		throw IOException("LLM HTTP " + std::to_string(status_code) + ": " + body);
	}

	// Parse response
	std::string error;
	std::string content;

	switch (prov) {
	case LlmProvider::LOCAL:
	case LlmProvider::OPENAI:
		content = ParseOpenAIResponse(body, error);
		break;
	case LlmProvider::GEMINI:
		content = ParseGeminiResponse(body, error);
		break;
	default:
		error = "Unsupported provider";
	}

	if (!error.empty()) {
		throw IOException("LLM error: " + error);
	}

	return content;
}

// Scalar function implementation
static void PromptScalarFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &context = state.GetContext();
	auto &prompt_vector = args.data[0];

	UnaryExecutor::Execute<string_t, string_t>(prompt_vector, result, args.size(), [&](string_t prompt) {
		std::string prompt_str = prompt.GetString();
		std::string response = ExecutePrompt(context, prompt_str);
		return StringVector::AddString(result, response);
	});
}

void RegisterPromptFunction(ExtensionLoader &loader) {
	auto prompt_func = ScalarFunction("prompt", {LogicalType::VARCHAR}, LogicalType::VARCHAR, PromptScalarFunction);
	loader.RegisterFunction(prompt_func);
}

} // namespace duckdb
