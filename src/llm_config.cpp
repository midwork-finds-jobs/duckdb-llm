#include "llm_extension.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/common/exception.hpp"

namespace duckdb {

void EnsureHttpRequestLoaded(Connection &conn);

// Helper function to get config value from DuckDB settings
std::string GetLlmConfig(ClientContext &context, const std::string &key) {
	Value result;
	if (context.TryGetCurrentSetting(Identifier(key), result)) {
		if (!result.IsNull()) {
			return result.ToString();
		}
	}
	return "";
}

void EnsureHttpRequestLoaded(Connection &conn) {
	if (DatabaseInstance::GetDatabase(*conn.context).ExtensionIsLoaded("http_request")) {
		return;
	}
	auto result = conn.Query("LOAD http_request");
	if (result->HasError()) {
		throw IOException("llm requires the http_request extension, which could not be loaded (%s). "
		                  "Install it with: INSTALL http_request FROM community",
		                  result->GetError());
	}
}

double GetLlmDefaultTemperature(ClientContext &context) {
	Value result;
	if (context.TryGetCurrentSetting(Identifier("llm_default_temperature"), result) && !result.IsNull()) {
		return result.DefaultCastAs(LogicalType::DOUBLE).GetValue<double>();
	}
	return 0.0;
}

int GetLlmDefaultMaxTokens(ClientContext &context) {
	Value result;
	if (context.TryGetCurrentSetting(Identifier("llm_default_max_tokens"), result) && !result.IsNull()) {
		return result.DefaultCastAs(LogicalType::INTEGER).GetValue<int>();
	}
	return 1024;
}

void RegisterLlmOptions(DatabaseInstance &db) {
	auto &config = DBConfig::GetConfig(db);

	// Local provider options
	config.AddExtensionOption("llm_base_url", "Base URL for local LLM provider", LogicalType::VARCHAR,
	                          Value("http://localhost:1234"));

	// OpenAI options
	config.AddExtensionOption("llm_openai_api_key", "OpenAI API key", LogicalType::VARCHAR, Value());
	config.AddExtensionOption("llm_openai_base_url", "OpenAI API base URL", LogicalType::VARCHAR,
	                          Value("https://api.openai.com"));

	// Gemini options
	config.AddExtensionOption("llm_gemini_api_key", "Gemini API key", LogicalType::VARCHAR, Value());

	// Cloudflare options
	config.AddExtensionOption("llm_cloudflare_api_key", "Cloudflare API token", LogicalType::VARCHAR, Value());
	config.AddExtensionOption("llm_cloudflare_account_id", "Cloudflare account ID", LogicalType::VARCHAR, Value());

	// Default settings
	config.AddExtensionOption("llm_default_provider", "Default LLM provider (local, openai, gemini, cloudflare)",
	                          LogicalType::VARCHAR, Value("local"));
	config.AddExtensionOption("llm_default_model", "Default model name", LogicalType::VARCHAR, Value());
	config.AddExtensionOption("llm_default_temperature", "Default temperature (0.0-1.0)", LogicalType::DOUBLE,
	                          Value(0.0));
	config.AddExtensionOption("llm_default_max_tokens", "Default max tokens", LogicalType::INTEGER, Value(1024));
}

} // namespace duckdb
