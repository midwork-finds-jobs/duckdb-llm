#pragma once

#include "duckdb.hpp"
#include <string>

namespace duckdb {

// Provider enumeration
enum class LlmProvider { LOCAL, OPENAI, GEMINI, CLOUDFLARE };

// Provider configuration
struct ProviderConfig {
	LlmProvider provider;
	std::string base_url;
	std::string model;
	std::string api_key;
	std::string account_id; // Cloudflare only
};

// Default values per provider
struct ProviderDefaults {
	static constexpr const char *LOCAL_URL = "http://localhost:1234";
	static constexpr const char *LOCAL_MODEL = "default";

	static constexpr const char *OPENAI_URL = "https://api.openai.com";
	static constexpr const char *OPENAI_MODEL = "gpt-4o-mini";

	static constexpr const char *GEMINI_URL = "https://generativelanguage.googleapis.com";
	static constexpr const char *GEMINI_MODEL = "gemini-2.5-flash-lite";

	static constexpr const char *CLOUDFLARE_URL = "https://api.cloudflare.com";
	static constexpr const char *CLOUDFLARE_MODEL = "@cf/meta/llama-4-scout-17b-16e-instruct";
};

// Build JSON request body for each provider
std::string BuildOpenAIRequest(const std::string &prompt, const std::string &model, double temperature, int max_tokens,
                               bool json_mode, const std::string &json_schema, const std::string &system_prompt,
                               const std::string &reasoning_effort);

std::string BuildGeminiRequest(const std::string &prompt, const std::string &model, double temperature, bool json_mode,
                               const std::string &json_schema, const std::string &system_prompt);

std::string BuildCloudflareRequest(const std::string &prompt, double temperature, int max_tokens,
                                   const std::string &system_prompt);

std::string BuildLocalRequest(const std::string &prompt, const std::string &model, double temperature, int max_tokens,
                              bool json_mode, const std::string &system_prompt);

// Build full URL for each provider
std::string BuildOpenAIUrl(const std::string &base_url);
std::string BuildGeminiUrl(const std::string &base_url, const std::string &model, const std::string &api_key);
std::string BuildCloudflareUrl(const std::string &base_url, const std::string &account_id, const std::string &model);
std::string BuildLocalUrl(const std::string &base_url);

// Parse response from each provider
std::string ParseOpenAIResponse(const std::string &response_body, std::string &error_out);
std::string ParseGeminiResponse(const std::string &response_body, std::string &error_out);
std::string ParseCloudflareResponse(const std::string &response_body, std::string &error_out);

// Parse provider string to enum
LlmProvider ParseProvider(const std::string &provider_str);
std::string ProviderToString(LlmProvider provider);

// Escape string for JSON
std::string EscapeJsonString(const std::string &str);

} // namespace duckdb
